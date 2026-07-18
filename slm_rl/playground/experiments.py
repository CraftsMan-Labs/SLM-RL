"""Experiment materialization + subprocess launch.

Read-WRITE surface (unlike slm_rl/webui/): each experiment is a directory
`<home>/playground/<name>/` this module creates itself —
`config/{default.yaml, games/<game>.yaml}` (the repo configs deep-merged
with the attendee's knob values), `reward_hook.py` (if reward code was
given), `experiment.json` (provenance), and `rollout.log` /
`evolve.log` (subprocess stdout). Nothing outside that directory is ever
written. Materializing the full config gives exact reproducibility for
free: the experiment dir IS its config (plan 013 design decision 2).

Stdlib-only: no gymnasium/ale_py/numpy imports (CODING_GUIDELINE 8GB rule).
The heavy work happens in a subprocess (`python -m slm_rl.cli rollout` /
`evolve`), never in this process.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from slm_rl.playground.knobs import KNOBS, PLAYGROUND_DEFAULT_OVERRIDES
from slm_rl.teachers.dqn_checkpoint import (
    expected_dqn_checkpoint,
    find_dqn_checkpoint,
    is_legacy_space_invaders_default,
)

_NAME_RE = re.compile(r"^[a-z0-9-]{1,40}$")

# Plan 022: playground model picker. `backend` choices mirror
# slm_rl.config.schema.Backend plus a "tier default" pseudo-choice meaning
# "don't set an override" (materializes to no `backend:` key at all, i.e.
# byte-identical to today). Kept as plain strings here (not imported from
# schema.Backend) so this stdlib-only module never imports pydantic.
BACKEND_CHOICES: tuple[str, ...] = (
    "tier default", "transformers", "transformers-4bit", "mlx",
)

# Subprocess log kinds written by launch_rollout / launch_evolve / launch_theater.
LOG_KINDS: frozenset[str] = frozenset({"rollout", "evolve", "theater"})
# ponytail: fixed 64KiB byte cap (not line-based); seek may land mid-UTF-8 char
# — decode with errors="replace". Real version would stream SSE or seek to a
# line boundary.
LOG_TAIL_BYTES: int = 64 * 1024
# Bake runs are long (DQN + hundreds of demos); keep a much larger tail so
# navigating away from Projects and back still shows recent history.
BAKE_LOG_TAIL_BYTES: int = 2 * 1024 * 1024


class Busy(Exception):
    """Raised when a subprocess of the requested kind is already running
    (plan 013 resource guard: at most 1 quick-experiment + 1 evolve
    subprocess at a time -- workshop laptops are weak, queues invite
    confusion). The server maps this to HTTP 409."""


class NotRunning(Exception):
    """Raised when stop is requested but no subprocess is active for that
    experiment. The server maps this to HTTP 404."""


class InvalidExperiment(Exception):
    """Bad name or reward code (syntax error) -- returned to the UI as a
    400, never written to disk."""


@dataclass
class ExperimentDir:
    name: str
    path: Path  # <home>/playground/<name>/
    run_id: str  # pg-<name>
    # Advisory model-id warnings from the create that produced this handle
    # (plan 022 design decision 2). Empty for handles built by callers other
    # than create_experiment (server.py's traversal-guard lookups, tests
    # constructing an ExperimentDir directly) -- those never ran validation
    # in the first place, so there is nothing to report.
    warnings: list[str] = field(default_factory=list)

    @property
    def config_dir(self) -> Path:
        return self.path / "config"

    @property
    def reward_hook_path(self) -> Path:
        return self.path / "reward_hook.py"

    @property
    def experiment_json(self) -> Path:
        return self.path / "experiment.json"

    def log_path(self, kind: str) -> Path:
        return self.path / f"{kind}.log"

    @property
    def run_dir(self) -> Path:
        """Where `RunPaths(home, run_id)` resolves for this experiment --
        the materialized default.yaml's `home` points here (see
        create_experiment), so `pg-<name>`'s generations/ live fully inside
        the experiment directory."""
        return self.path / self.run_id


# One lock per subprocess kind (plan 013 design decision 6): a workshop
# laptop can't usefully run two heavy rollouts at once, and a queue would
# just confuse attendees about which experiment is running. "theater" (plan
# 020) gets its own kind: an exhibition is a separate CPU-bound subprocess
# from quick/evolve and attendees may reasonably want to launch one right
# after an evolve finishes, without waiting on the quick-experiment lock.
# "bake" is instructor-only pack baking (UI), same single-flight rule.
_LOCKS: dict[str, threading.Lock] = {
    "quick": threading.Lock(),
    "evolve": threading.Lock(),
    "theater": threading.Lock(),
    "bake": threading.Lock(),
}
_ACTIVE: dict[str, subprocess.Popen | None] = {
    "quick": None, "evolve": None, "theater": None, "bake": None,
}
# Experiment name that owns each kind (bake has no owner). Cleared with
# `_ACTIVE` when the process exits so stop can target one experiment.
_ACTIVE_OWNER: dict[str, str | None] = {
    "quick": None, "evolve": None, "theater": None, "bake": None,
}
_JOB_KINDS: tuple[str, ...] = ("quick", "evolve", "theater")
_STATE_LOCK = threading.Lock()
_STOP_GRACE_SECONDS = 5.0


def _busy(kind: str) -> bool:
    with _STATE_LOCK:
        proc = _ACTIVE.get(kind)
        if proc is None:
            return False
        if proc.poll() is None:
            return True
        _ACTIVE[kind] = None
        _ACTIVE_OWNER[kind] = None
        return False


def _mark_active(
    kind: str, proc: subprocess.Popen, *, owner: str | None = None,
) -> None:
    with _STATE_LOCK:
        _ACTIVE[kind] = proc
        _ACTIVE_OWNER[kind] = owner


def active_jobs_for(name: str) -> list[str]:
    """Kinds currently running for experiment `name` (may be empty)."""
    live: list[str] = []
    for kind in _JOB_KINDS:
        with _STATE_LOCK:
            proc = _ACTIVE.get(kind)
            owner = _ACTIVE_OWNER.get(kind)
            if proc is None or owner != name:
                continue
            if proc.poll() is None:
                live.append(kind)
            else:
                _ACTIVE[kind] = None
                _ACTIVE_OWNER[kind] = None
    return live


def _terminate_proc(proc: subprocess.Popen) -> None:
    """SIGTERM the process group (or the process), then SIGKILL if needed."""
    if proc.poll() is not None:
        return
    pid = proc.pid
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except ProcessLookupError:
            return
    deadline = time.monotonic() + _STOP_GRACE_SECONDS
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass


def stop_experiment(name: str, kinds: list[str] | tuple[str, ...] | None = None) -> list[str]:
    """Kill active quick/evolve/theater subprocesses owned by `name`.

    `kinds` limits which job types to stop (e.g. ``("theater",)`` so evolve
    keeps collecting rollouts). Default ``None`` stops every active kind for
    this experiment — same as the original Stop button.

    Returns the kinds that were signaled. Raises NotRunning when none are
    active for this experiment (so the UI can distinguish "already idle"
    from a successful stop). Bake is instructor-global and is not stopped
    here.
    """
    wanted = tuple(_JOB_KINDS) if kinds is None else tuple(kinds)
    unknown = [k for k in wanted if k not in _JOB_KINDS]
    if unknown:
        raise ValueError(
            f"unknown stop kind(s) {unknown!r}: must be one of {list(_JOB_KINDS)}"
        )
    stopped: list[str] = []
    for kind in wanted:
        with _STATE_LOCK:
            proc = _ACTIVE.get(kind)
            owner = _ACTIVE_OWNER.get(kind)
            if proc is None or owner != name:
                continue
            if proc.poll() is not None:
                _ACTIVE[kind] = None
                _ACTIVE_OWNER[kind] = None
                continue
            target = proc
        _terminate_proc(target)
        with _STATE_LOCK:
            if _ACTIVE.get(kind) is target:
                _ACTIVE[kind] = None
                _ACTIVE_OWNER[kind] = None
        stopped.append(kind)
    if not stopped:
        raise NotRunning(f"no active job for experiment {name!r}")
    return stopped


def validate_name(name: str) -> None:
    if not _NAME_RE.match(name):
        raise InvalidExperiment(
            f"invalid experiment name {name!r}: must match [a-z0-9-]{{1,40}}"
        )


def tail_log(exp: ExperimentDir, kind: str, max_bytes: int = LOG_TAIL_BYTES) -> str:
    """Last `max_bytes` of `<exp>/<kind>.log` as text. Read-only.

    Returns "" if the log file does not exist yet (experiment created but
    subprocess has not written). Caller must validate `kind` ∈ LOG_KINDS.
    Never reads profile / tokens — only the subprocess stdout redirect.
    """
    path = exp.log_path(kind)
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
        data = f.read()
    return data.decode("utf-8", errors="replace")


def _validate_model_id_local(model: str) -> None:
    """Blocking local sanity checks only (design decision 2): whitespace, or
    missing '/' unless it's a local path that already exists on disk (a
    local HF snapshot dir — a legal `model_id` for transformers). Never a
    network call."""
    if model != model.strip() or not model:
        raise InvalidExperiment(f"invalid model id {model!r}: leading/trailing whitespace or empty")
    if "/" not in model and not Path(model).exists():
        raise InvalidExperiment(
            f"invalid model id {model!r}: expected 'org/repo' (HF hub id) or an existing local path"
        )


_KNOB_TARGETS: dict[str, str] = {knob.key: knob.target for knob in KNOBS}


def _apply_knob(
    run_data: dict[str, Any],
    game_data: dict[str, Any],
    key: str,
    value: Any,
    *,
    game: str,
    home: Path,
) -> None:
    target = _KNOB_TARGETS.get(key)
    if target is None:
        raise InvalidExperiment(f"unknown knob: {key!r}")

    # `teacher` is resolved in `_resolve_teacher` (needs optional HF URL).
    if key == "teacher":
        return

    if target == "game":
        game_data[key] = value
    elif target == "game.monitor":
        game_data.setdefault("monitor", {})[key] = value
    elif target == "game.extra":
        game_data.setdefault("extra", {})[key] = value
    elif target == "run.train":
        run_data.setdefault("train", {})[key] = value
    elif target == "run.teacher":
        run_data.setdefault("teacher", {})[key] = value
    else:  # pragma: no cover -- exhaustive over KnobTarget
        raise InvalidExperiment(f"unhandled knob target: {target!r}")


def _resolve_teacher(
    run_data: dict[str, Any],
    *,
    game: str,
    home: Path,
    teacher: Any,
    dqn_url: str | None,
) -> str | None:
    """Materialize `teacher.dqn_checkpoint` from local bake and/or HF URL.

    Returns the normalized HF repo id when a URL was used, else None.
    """
    run_data.setdefault("teacher", {})
    if teacher is None:
        return None

    url = (dqn_url or "").strip() or None
    normalized_url: str | None = None
    if url:
        from slm_rl.packs import normalize_repo_id

        try:
            normalized_url = normalize_repo_id(url)
        except ValueError as exc:
            raise InvalidExperiment(str(exc)) from exc

    if teacher == "heuristic":
        # Heuristic for evolve/SFT, but keep dqn_url so Quick Run can still
        # load the workshop Hugging Face DQN (never trains a fresh network).
        run_data["teacher"]["dqn_checkpoint"] = None
        if normalized_url:
            run_data["dqn_url"] = normalized_url
        return normalized_url
    if teacher != "dqn":
        raise InvalidExperiment(f"unknown teacher choice: {teacher!r}")

    if normalized_url:
        from slm_rl.hf_auth import apply_hf_token
        from slm_rl.packs import resolve_dqn
        from slm_rl.playground.profile import load_profile

        # Parent may lack HF_TOKEN; profile.json is the workshop source of truth.
        profile = load_profile(home)
        apply_hf_token(profile.hf_token if profile else None)
        try:
            pt = resolve_dqn(normalized_url, home, game)
        except ValueError as exc:
            raise InvalidExperiment(str(exc)) from exc
        run_data["teacher"]["dqn_checkpoint"] = str(pt.resolve())
        run_data["dqn_url"] = normalized_url
        return normalized_url

    found = find_dqn_checkpoint(game, home)
    if found is not None:
        run_data["teacher"]["dqn_checkpoint"] = str(found)
        return None

    expected = expected_dqn_checkpoint(game, home)
    raise InvalidExperiment(
        f"Teacher is DQN but no checkpoint for {game!r}. "
        f"Paste a Hugging Face repo (org/name) that contains dqn.pt, "
        f"or bake a workshop pack for this game first.\n"
        f"  Expected local path: {expected}"
    )


def create_experiment(
    home: Path | str,
    game: str,
    name: str,
    knob_values: dict[str, Any],
    reward_code: str | None = None,
    model: str | None = None,
    backend: str | None = None,
    dqn_url: str | None = None,
    dataset_url: str | None = None,
    adapter_url: str | None = None,
) -> ExperimentDir:
    """Materialize `<home>/playground/<name>/` from the repo configs +
    `knob_values`, plus `reward_hook.py` if `reward_code` is given (after a
    `compile()` check -- syntax errors raise InvalidExperiment, nothing is
    written). Overwrites a previous experiment of the same name (re-running
    with tweaked knobs is the expected workshop loop).

    `model` (free-text HF repo id or local path) and `backend` (one of
    BACKEND_CHOICES) are dedicated form fields (plan 022 design decision 1),
    not knobs -- KnobType has no free-string variant, same reason `opponent`
    was skipped as a knob in plan 019. Both are optional; omitting either
    leaves the corresponding key out of the materialized config entirely,
    so `RunConfig.model`/`backend` fall back to the tier default exactly as
    they do today (byte-identical when unset -- hard rule 3).

    `dqn_url` (optional HF dataset/model repo containing `dqn.pt`) is used
    when `knob_values["teacher"] == "dqn"`: download into
    `<home>/packs/` and point `teacher.dqn_checkpoint` at the cached file.
    A local bake pack still works with an empty URL.

    `dataset_url` (optional public HF dataset pack id) is written into the
    materialized run config so Evolve can warm-start from the pack without
    re-pasting the URL.

    `adapter_url` (optional public HF *model* repo with `adapter/`) is written
    so Evolve can import a published SFT LoRA as gen-1 champion and skip re-SFT.

    The returned `ExperimentDir.warnings` carries non-blocking advisory text
    (a model id we couldn't verify against the Hub, offline or not) -- the
    experiment is always created regardless of what it says (design
    decision 2). Blocking validation (whitespace, missing '/' on an id that
    isn't also a local path, or an unrecognized backend name) raises
    InvalidExperiment instead, same as a bad name or bad reward code.
    """
    from slm_rl.config.loader import CONFIG_DIR, load_yaml

    validate_name(name)

    if reward_code is not None and reward_code.strip():
        try:
            compile(reward_code, "<reward_hook>", "exec")
        except SyntaxError as exc:
            raise InvalidExperiment(f"reward code has a syntax error: {exc}") from exc

    warnings: list[str] = []
    if model is not None and model != "":
        # Validate the RAW value first (whitespace is itself the blocking
        # condition -- stripping before the check would make it unreachable).
        _validate_model_id_local(model)
        model = model.strip()
    else:
        model = None

    backend = backend if backend and backend != "tier default" else None
    if backend is not None and backend not in BACKEND_CHOICES:
        raise InvalidExperiment(f"unknown backend {backend!r}: must be one of {BACKEND_CHOICES}")

    home = Path(home)
    exp = ExperimentDir(name=name, path=home / "playground" / name, run_id=f"pg-{name}", warnings=warnings)
    exp.config_dir.mkdir(parents=True, exist_ok=True)
    (exp.config_dir / "games").mkdir(parents=True, exist_ok=True)

    run_data = load_yaml(CONFIG_DIR / "default.yaml")
    game_path = CONFIG_DIR / "games" / f"{game}.yaml"
    game_data = load_yaml(game_path) if game_path.exists() else {}

    # Workshop defaults (e.g. episodes_per_generation=2) when the form omits
    # a key — still overridable via explicit knob_values.
    merged_knobs = {**PLAYGROUND_DEFAULT_OVERRIDES, **(knob_values or {})}
    for key, value in merged_knobs.items():
        _apply_knob(run_data, game_data, key, value, game=game, home=home)

    resolved_dqn_url = _resolve_teacher(
        run_data,
        game=game,
        home=home,
        teacher=knob_values.get("teacher"),
        dqn_url=dqn_url,
    )

    resolved_dataset_url = (dataset_url or "").strip() or None
    if resolved_dataset_url:
        from slm_rl.packs import normalize_repo_id

        try:
            resolved_dataset_url = normalize_repo_id(resolved_dataset_url)
        except ValueError as exc:
            raise InvalidExperiment(str(exc)) from exc
        run_data["dataset_url"] = resolved_dataset_url

    resolved_adapter_url = (adapter_url or "").strip() or None
    if resolved_adapter_url:
        from slm_rl.packs import normalize_repo_id

        try:
            resolved_adapter_url = normalize_repo_id(resolved_adapter_url)
        except ValueError as exc:
            raise InvalidExperiment(str(exc)) from exc
        run_data["adapter_url"] = resolved_adapter_url

    run_data["game"] = game
    run_data["home"] = str(exp.path)
    run_data["run_id"] = exp.run_id
    if model:
        run_data["model"] = model
    if backend:
        run_data["backend"] = backend

    if reward_code is not None and reward_code.strip():
        exp.reward_hook_path.write_text(reward_code, encoding="utf-8")
        game_data.setdefault("extra", {})["reward_hook"] = str(exp.reward_hook_path.resolve())

    with (exp.config_dir / "default.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(run_data, f, sort_keys=False)
    with (exp.config_dir / "games" / f"{game}.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(game_data, f, sort_keys=False)

    exp.experiment_json.write_text(
        json.dumps(
            {
                "name": name,
                "game": game,
                "knob_values": merged_knobs,
                "model": model,
                "backend": backend,
                "dqn_url": resolved_dqn_url,
                "dataset_url": resolved_dataset_url,
                "adapter_url": resolved_adapter_url,
                "warnings": warnings,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return exp


def update_experiment_knobs(
    exp: ExperimentDir,
    knob_values: dict[str, Any],
) -> dict[str, Any]:
    """Patch knobs on an existing experiment (rewrites config + provenance).

    Used when attendees lower ``episodes_per_generation`` mid-workshop without
    recreating the project. Does not touch model/backend/URLs. Raises
    InvalidExperiment on unknown keys. Returns the merged knob_values.
    """
    if not knob_values:
        raise InvalidExperiment("no knob_values provided")
    try:
        run_data = yaml.safe_load(
            (exp.config_dir / "default.yaml").read_text(encoding="utf-8")
        ) or {}
    except OSError as exc:
        raise InvalidExperiment(f"missing experiment config: {exc}") from exc
    game = run_data.get("game")
    if not game:
        try:
            meta = json.loads(exp.experiment_json.read_text(encoding="utf-8"))
            game = meta.get("game")
        except (OSError, json.JSONDecodeError):
            game = None
    if not game:
        raise InvalidExperiment("experiment has no game")
    game_path = exp.config_dir / "games" / f"{game}.yaml"
    try:
        game_data = yaml.safe_load(game_path.read_text(encoding="utf-8")) or {}
    except OSError:
        game_data = {}

    # <runs>/playground/<name> → <runs> (packs / teacher resolution root).
    home_for_knobs = exp.path.parent.parent

    for key, value in knob_values.items():
        _apply_knob(
            run_data, game_data, key, value, game=game, home=home_for_knobs,
        )

    with (exp.config_dir / "default.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(run_data, f, sort_keys=False)
    game_path.parent.mkdir(parents=True, exist_ok=True)
    with game_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(game_data, f, sort_keys=False)

    try:
        meta = json.loads(exp.experiment_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        meta = {"name": exp.name, "game": game, "knob_values": {}}
    merged = {**(meta.get("knob_values") or {}), **knob_values}
    meta["knob_values"] = merged
    exp.experiment_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Keep run_config.yaml in sync when evolve already wrote one.
    run_cfg = exp.run_dir / "run_config.yaml"
    if run_cfg.is_file():
        try:
            rc = yaml.safe_load(run_cfg.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            rc = {}
        for key, value in knob_values.items():
            target = _KNOB_TARGETS.get(key)
            if target == "run.train":
                rc.setdefault("train", {})[key] = value
            elif target == "run.teacher" and key != "teacher":
                rc.setdefault("teacher", {})[key] = value
        with run_cfg.open("w", encoding="utf-8") as f:
            yaml.safe_dump(rc, f, sort_keys=False)

    return merged


def _experiment_meta(exp: ExperimentDir) -> dict[str, Any]:
    try:
        return json.loads(exp.experiment_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _materialized_dqn_checkpoint(
    exp: ExperimentDir, *, token: str | None = None,
) -> str | None:
    """Resolve the DQN checkpoint for a quick-run (`rollout --agent solver`).

    The `rollout` CLI only honors ``--dqn-checkpoint`` (it does not read
    ``run.teacher.dqn_checkpoint``), so Quick Run must thread the path here.

    Resolution order (never trains a fresh DQN):
      1. ``teacher.dqn_checkpoint`` if the file exists
      2. ``dqn_url`` / experiment provenance → download via ``resolve_dqn``
         into ``<home>/packs/`` and persist the path (workshop HF DQN)
      3. Local bake pack under ``<home>/packs/<game>/dqn.pt``
      4. Else ``None`` → ``make_teacher`` falls back to the game heuristic

    Quick Run never creates a new DQN. If ``dqn_url`` is set but download
    fails, raises ``InvalidExperiment`` (loud — do not silently heuristic).
    """
    default_yaml = exp.config_dir / "default.yaml"
    if not default_yaml.exists():
        return None
    data = yaml.safe_load(default_yaml.read_text(encoding="utf-8")) or {}
    meta = _experiment_meta(exp)
    home = exp.path.parent.parent  # <home>/playground/<name> → <home>
    game = data.get("game") if isinstance(data.get("game"), str) else meta.get("game")

    configured = data.get("teacher", {}).get("dqn_checkpoint")
    if configured and Path(configured).is_file():
        return str(Path(configured).resolve())

    dqn_url = (data.get("dqn_url") or meta.get("dqn_url") or "").strip() or None
    if dqn_url and isinstance(game, str) and game:
        from slm_rl.packs import normalize_repo_id, resolve_dqn

        repo = normalize_repo_id(dqn_url)
        # Parent process may lack HF_TOKEN (token lives in profile.json).
        overlay = _hf_child_env(token) or {}
        saved = {k: os.environ.get(k) for k in overlay}
        try:
            os.environ.update(overlay)
            pt = resolve_dqn(repo, home, game)
        except ValueError as exc:
            raise InvalidExperiment(
                f"dqn_url {repo!r} configured but dqn.pt could not be loaded: {exc}"
            ) from exc
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        path = str(Path(pt).resolve())
        data.setdefault("teacher", {})["dqn_checkpoint"] = path
        data["dqn_url"] = repo
        try:
            default_yaml.write_text(
                yaml.safe_dump(data, sort_keys=False), encoding="utf-8",
            )
        except OSError:
            pass
        return path

    if isinstance(game, str) and game:
        found = find_dqn_checkpoint(game, home)
        if found is not None:
            return str(found)
        if configured and is_legacy_space_invaders_default(configured, game):
            teacher = data.setdefault("teacher", {})
            teacher["dqn_checkpoint"] = None
            try:
                default_yaml.write_text(
                    yaml.safe_dump(data, sort_keys=False), encoding="utf-8",
                )
            except OSError:
                pass
            return None

    return None


def _hf_child_env(token: str | None) -> dict[str, str] | None:
    """Env overlay so evolve/bake/theater Hub calls use the welcome-screen token.

    Workshop tokens live in `profile.json`, not the Docker/`os.environ` of the
    API process. Subprocesses must inherit an explicit HF_TOKEN or transformers /
    huggingface_hub fall back to anonymous downloads (rate limits + the
    "unauthenticated requests" warning).
    """
    from slm_rl.hf_auth import hf_token

    value = (token or "").strip() or hf_token()
    if not value:
        return None
    return {
        "HF_TOKEN": value,
        "HUGGING_FACE_HUB_TOKEN": value,
    }


def _spawn(
    cmd: list[str],
    log_path: Path,
    env: dict[str, str] | None = None,
    *,
    append: bool = False,
) -> subprocess.Popen:
    # Close the parent's copy of the log handle once Popen returns -- the
    # child holds its own dup of the fd, so the parent copy would otherwise
    # leak one open file per launched experiment for the server's lifetime.
    # start_new_session=True puts the child in its own process group so
    # stop_experiment can SIGTERM/SIGKILL the whole tree (trainer workers).
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_env = None
    if env is not None:
        child_env = {**os.environ, **env}
    mode = "a" if append else "w"
    with open(log_path, mode, encoding="utf-8") as log_file:
        return subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=child_env,
            start_new_session=True,
        )


def _launch_exclusive(
    kind: str, busy_msg: str, cmd: list[str], log_path: Path,
    env: dict[str, str] | None = None,
    *,
    owner: str | None = None,
    append: bool = False,
) -> subprocess.Popen:
    """Acquire the kind's lock, spawn `cmd` with stdout→`log_path`, mark
    active. Raises Busy if another subprocess of this kind is running."""
    if not _LOCKS[kind].acquire(blocking=False):
        raise Busy(busy_msg)
    try:
        if _busy(kind):
            raise Busy(busy_msg)
        proc = _spawn(cmd, log_path, env=env, append=append)
        _mark_active(kind, proc, owner=owner)
        # Sidecar PID so monitors can tell "log says rollout" from "process
        # still alive" after API restarts clear the in-memory job table.
        try:
            log_path.with_suffix(".pid").write_text(f"{proc.pid}\n", encoding="utf-8")
        except OSError:
            pass
        return proc
    finally:
        _LOCKS[kind].release()


def bake_log_path(home: Path | str) -> Path:
    return Path(home) / "packs" / "bake.log"


def tail_bake_log(home: Path | str, max_bytes: int = BAKE_LOG_TAIL_BYTES) -> str:
    path = bake_log_path(home)
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
        data = f.read()
    text = data.decode("utf-8", errors="replace")
    if size > max_bytes:
        text = f"…(earlier bake history truncated; full log at {path})\n" + text
    return text


def is_bake_busy() -> bool:
    return _busy("bake")


def launch_bake(
    home: Path | str,
    *,
    game: str | None = None,
    all_games: bool = False,
    episodes: int = 1000,
    dqn_decisions: int = 50_000,
    device: str = "cpu",
    seed: int = 0,
    push: str | None = None,
    push_prefix: str | None = None,
    token: str | None = None,
    selection_quantile: float = 0.25,
) -> subprocess.Popen:
    """UI bake: spawn `python -m slm_rl.packs` (not a user CLI).

    Appends to `packs/bake.log` (never truncates) so navigating away from
    the Projects page and returning still shows prior bake output.
    """
    from datetime import datetime, timezone

    if not all_games and not game:
        raise InvalidExperiment("pick a game or bake all games")
    if not 0.0 < selection_quantile <= 1.0:
        raise InvalidExperiment(
            f"selection_quantile must be in (0, 1], got {selection_quantile}"
        )
    home = Path(home)
    cmd = [
        sys.executable, "-m", "slm_rl.packs",
        "--home", str(home),
        "--episodes", str(episodes),
        "--dqn-decisions", str(dqn_decisions),
        "--selection-quantile", str(selection_quantile),
        "--device", device,
        "--seed", str(seed),
    ]
    if all_games:
        cmd.append("--all")
    else:
        cmd.extend(["--game", game])  # type: ignore[arg-type]
    if push:
        cmd.extend(["--push", push])
    if push_prefix:
        cmd.extend(["--push-prefix", push_prefix])
    env = _hf_child_env(token)
    log_path = bake_log_path(home)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    label = "all" if all_games else (game or "?")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with log_path.open("a", encoding="utf-8") as header:
        header.write(
            f"\n========== bake {stamp} game={label} "
            f"episodes={episodes} dqn_decisions={dqn_decisions} "
            f"selection_quantile={selection_quantile} ==========\n"
        )
    return _launch_exclusive(
        "bake",
        "Busy: a pack bake is already running — wait for it to finish",
        cmd,
        log_path,
        env=env,
        append=True,
    )


def launch_rollout(
    exp: ExperimentDir,
    game: str,
    agent: str,
    episodes: int,
    seed: int,
    token: str | None = None,
) -> subprocess.Popen:
    """Launch a quick CPU experiment (`--agent solver|random` only -- never
    `llm`, this is a fast screening loop, not the real training pipeline).
    Raises Busy if another quick experiment is already running.

    When the experiment has ``dqn_url`` (or a materialized checkpoint), the
    HF/baked DQN is passed as ``--dqn-checkpoint``. Never trains a new DQN.
    """
    cmd = [
        sys.executable, "-m", "slm_rl.cli", "rollout",
        "--game", game,
        "--agent", agent,
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--run-id", exp.run_id,
        "--config-dir", str(exp.config_dir),
    ]
    dqn_checkpoint = _materialized_dqn_checkpoint(exp, token=token)
    if dqn_checkpoint:
        cmd += ["--dqn-checkpoint", dqn_checkpoint]
    return _launch_exclusive(
        "quick",
        "Busy: a quick experiment is already running — wait for it to finish (only one at a time)",
        cmd, exp.log_path("rollout"),
        env=_hf_child_env(token),
        owner=exp.name,
    )


def launch_evolve(
    exp: ExperimentDir,
    game: str,
    generations: int,
    dataset_url: str | None = None,
    dqn_url: str | None = None,
    adapter_url: str | None = None,
    token: str | None = None,
) -> subprocess.Popen:
    """Launch the real rollout->train->eval loop for a config that survived
    the quick screen. Raises Busy if another evolve is already running.

    `token` is the attendee HF token from the welcome screen — forwarded as
    HF_TOKEN so base-model / pack Hub downloads are authenticated.
    """
    cmd = [
        sys.executable, "-m", "slm_rl.cli", "evolve",
        "--game", game,
        "--generations", str(generations),
        "--run-id", exp.run_id,
        "--config-dir", str(exp.config_dir),
        "--warm-start",
    ]
    if dataset_url:
        cmd.extend(["--dataset-url", dataset_url])
    if dqn_url:
        cmd.extend(["--dqn-url", dqn_url])
    if adapter_url:
        cmd.extend(["--adapter-url", adapter_url])
    # Workshop pack / published SFT: skip gen-0 frozen baseline. That path
    # loads the base LLM and plays ~100 eval episodes before any rollouts
    # land — the live watch panel stays on "Waiting for episodes" the whole
    # time. DIY (no URLs) still measures the stock model honestly.
    if dataset_url or adapter_url:
        cmd.append("--skip-baseline")
    return _launch_exclusive(
        "evolve",
        "Busy: an evolve run is already in progress — wait for it to finish (only one at a time)",
        cmd, exp.log_path("evolve"),
        env=_hf_child_env(token),
        owner=exp.name,
    )


def launch_theater(
    exp: ExperimentDir,
    game: str,
    episodes: int = 10,
    seed_start: int = 20_000,
    token: str | None = None,
) -> subprocess.Popen:
    """Launch `slm-rl theater` against this experiment's own run dir (plan
    020) -- base-vs-champion exhibition. Raises Busy if another exhibition
    is already running. The CLI itself requires `run_config.yaml` (written
    by `GenerationRunner.__init__`, i.e. only after `evolve` has run at
    least once) and exits loudly if it's missing -- an experiment that only
    ever ran a quick screen has no champion to exhibit yet."""
    cmd = [
        sys.executable, "-m", "slm_rl.cli", "theater",
        "--run-id", exp.run_id,
        # exp.path, not the playground root: create_experiment writes
        # `home: <exp.path>` into the materialized default.yaml (so
        # RunPaths(exp.path, exp.run_id).root == exp.run_dir), and the
        # theater CLI needs that SAME home to find run_config.yaml/
        # registry.json/adapters -- the playground root itself has no
        # run at its top level.
        "--home", str(exp.path),
        "--game", game,
        "--episodes", str(episodes),
        "--seed-start", str(seed_start),
        "--config-dir", str(exp.config_dir),
    ]
    return _launch_exclusive(
        "theater",
        "Busy: an exhibition is already running — wait for it to finish (only one at a time)",
        cmd, exp.log_path("theater"),
        env=_hf_child_env(token),
        owner=exp.name,
    )


def resolve_play_again_generation(
    exp: ExperimentDir, *, gen: int | None, champion: bool,
) -> int:
    """Pick the checkpoint generation for play-again (plan 026 Phase G).

    `champion=True` reads `registry.json` like theater/exhibition does.
    Otherwise `gen` must be a non-negative int. Raises InvalidExperiment on
    missing registry / no promotion / bad args.
    """
    if champion:
        registry_path = exp.run_dir / "registry.json"
        if not registry_path.exists():
            raise InvalidExperiment(
                "no registry.json — evolve at least once before play-again"
            )
        from slm_rl.orchestrator.registry import ModelRegistry

        champ = ModelRegistry(registry_path).champion
        if champ <= 0:
            raise InvalidExperiment(
                f"no promoted champion yet (registry champion={champ})"
            )
        return champ
    if gen is None:
        raise InvalidExperiment("provide gen (int) or champion=true")
    if not isinstance(gen, int) or isinstance(gen, bool) or gen < 0:
        raise InvalidExperiment(f"gen must be a non-negative int, got {gen!r}")
    return gen


def launch_play_again(
    exp: ExperimentDir,
    game: str,
    *,
    generation: int,
    episodes: int = 10,
    seed: int = 20_000,
    temperature: float = 0.2,
) -> subprocess.Popen:
    """Launch `slm-rl play-again` for one checkpoint into theater/play/.

    Shares the theater single-flight lock (plan 013: one exhibition-like
    subprocess at a time). Writes theater.log. No new trainer — CLI calls
    exhibition.run_play_again only.
    """
    if not (exp.run_dir / "run_config.yaml").exists():
        raise InvalidExperiment(
            "no run_config.yaml — evolve at least once before play-again"
        )
    if generation > 0:
        from slm_rl.orchestrator.paths import RunPaths

        adapter = RunPaths(exp.path, exp.run_id).adapter(generation)
        if not adapter.exists():
            raise InvalidExperiment(
                f"no adapter for generation {generation} at {adapter}"
            )
    if episodes < 1:
        raise InvalidExperiment(f"episodes must be >= 1, got {episodes}")
    if not (0.0 <= temperature <= 2.0):
        raise InvalidExperiment(
            f"temperature must be in [0, 2], got {temperature}"
        )

    cmd = [
        sys.executable, "-m", "slm_rl.cli", "play-again",
        "--run-id", exp.run_id,
        "--home", str(exp.path),
        "--game", game,
        "--generation", str(generation),
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--temperature", str(temperature),
        "--config-dir", str(exp.config_dir),
    ]
    return _launch_exclusive(
        "theater",
        "Busy: an exhibition is already running — wait for it to finish (only one at a time)",
        cmd, exp.log_path("theater"),
        owner=exp.name,
    )


