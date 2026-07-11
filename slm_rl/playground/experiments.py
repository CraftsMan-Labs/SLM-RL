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
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from slm_rl.playground.knobs import DEFAULT_DQN_CHECKPOINT, KNOBS

_NAME_RE = re.compile(r"^[a-z0-9-]{1,40}$")


class Busy(Exception):
    """Raised when a subprocess of the requested kind is already running
    (plan 013 resource guard: at most 1 quick-experiment + 1 evolve
    subprocess at a time -- workshop laptops are weak, queues invite
    confusion). The server maps this to HTTP 409."""


class InvalidExperiment(Exception):
    """Bad name or reward code (syntax error) -- returned to the UI as a
    400, never written to disk."""


@dataclass
class ExperimentDir:
    name: str
    path: Path  # <home>/playground/<name>/
    run_id: str  # pg-<name>

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
# just confuse attendees about which experiment is running.
_LOCKS: dict[str, threading.Lock] = {"quick": threading.Lock(), "evolve": threading.Lock()}
_ACTIVE: dict[str, subprocess.Popen | None] = {"quick": None, "evolve": None}
_STATE_LOCK = threading.Lock()


def _busy(kind: str) -> bool:
    with _STATE_LOCK:
        proc = _ACTIVE.get(kind)
        if proc is None:
            return False
        if proc.poll() is None:
            return True
        _ACTIVE[kind] = None
        return False


def _mark_active(kind: str, proc: subprocess.Popen) -> None:
    with _STATE_LOCK:
        _ACTIVE[kind] = proc


def validate_name(name: str) -> None:
    if not _NAME_RE.match(name):
        raise InvalidExperiment(
            f"invalid experiment name {name!r}: must match [a-z0-9-]{{1,40}}"
        )


def _knob_targets() -> dict[str, str]:
    return {knob.key: knob.target for knob in KNOBS}


def _apply_knob(run_data: dict[str, Any], game_data: dict[str, Any], key: str, value: Any) -> None:
    target = _knob_targets().get(key)
    if target is None:
        raise InvalidExperiment(f"unknown knob: {key!r}")

    if key == "teacher":
        run_data.setdefault("teacher", {})
        if value == "dqn":
            run_data["teacher"]["dqn_checkpoint"] = DEFAULT_DQN_CHECKPOINT
        elif value == "heuristic":
            run_data["teacher"]["dqn_checkpoint"] = None
        else:
            raise InvalidExperiment(f"unknown teacher choice: {value!r}")
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


def create_experiment(
    home: Path | str,
    game: str,
    name: str,
    knob_values: dict[str, Any],
    reward_code: str | None = None,
) -> ExperimentDir:
    """Materialize `<home>/playground/<name>/` from the repo configs +
    `knob_values`, plus `reward_hook.py` if `reward_code` is given (after a
    `compile()` check -- syntax errors raise InvalidExperiment, nothing is
    written). Overwrites a previous experiment of the same name (re-running
    with tweaked knobs is the expected workshop loop)."""
    from slm_rl.config.loader import CONFIG_DIR, load_yaml

    validate_name(name)

    if reward_code is not None and reward_code.strip():
        try:
            compile(reward_code, "<reward_hook>", "exec")
        except SyntaxError as exc:
            raise InvalidExperiment(f"reward code has a syntax error: {exc}") from exc

    home = Path(home)
    exp = ExperimentDir(name=name, path=home / "playground" / name, run_id=f"pg-{name}")
    exp.config_dir.mkdir(parents=True, exist_ok=True)
    (exp.config_dir / "games").mkdir(parents=True, exist_ok=True)

    run_data = load_yaml(CONFIG_DIR / "default.yaml")
    game_path = CONFIG_DIR / "games" / f"{game}.yaml"
    game_data = load_yaml(game_path) if game_path.exists() else {}

    for key, value in knob_values.items():
        _apply_knob(run_data, game_data, key, value)

    run_data["game"] = game
    run_data["home"] = str(exp.path)
    run_data["run_id"] = exp.run_id

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
                "knob_values": knob_values,
                "has_reward_code": bool(reward_code and reward_code.strip()),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "git_sha": _git_sha(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return exp


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except Exception:
        return None


def _materialized_dqn_checkpoint(exp: ExperimentDir) -> str | None:
    """Read `teacher.dqn_checkpoint` back out of the experiment's own
    materialized default.yaml. The `rollout` CLI command does NOT read
    run_cfg.teacher.dqn_checkpoint itself (it only wires up its own
    --dqn-checkpoint flag, see cli.py) -- so the "dqn" teacher knob is a
    no-op for the quick-experiment subprocess unless we thread it through
    explicitly here."""
    default_yaml = exp.config_dir / "default.yaml"
    if not default_yaml.exists():
        return None
    data = yaml.safe_load(default_yaml.read_text(encoding="utf-8")) or {}
    return data.get("teacher", {}).get("dqn_checkpoint")


def launch_rollout(
    exp: ExperimentDir,
    game: str,
    agent: str,
    episodes: int,
    seed: int,
) -> subprocess.Popen:
    """Launch a quick CPU experiment (`--agent solver|random` only -- never
    `llm`, this is a fast screening loop, not the real training pipeline).
    Raises Busy if another quick experiment is already running."""
    if not _LOCKS["quick"].acquire(blocking=False):
        raise Busy("a quick experiment is already running")
    try:
        if _busy("quick"):
            raise Busy("a quick experiment is already running")
        cmd = [
            sys.executable, "-m", "slm_rl.cli", "rollout",
            "--game", game,
            "--agent", agent,
            "--episodes", str(episodes),
            "--seed", str(seed),
            "--run-id", exp.run_id,
            "--config-dir", str(exp.config_dir),
        ]
        dqn_checkpoint = _materialized_dqn_checkpoint(exp)
        if dqn_checkpoint:
            cmd += ["--dqn-checkpoint", dqn_checkpoint]
        log = exp.log_path("rollout")
        proc = _spawn(cmd, log)
        _mark_active("quick", proc)
        return proc
    finally:
        _LOCKS["quick"].release()


def launch_evolve(exp: ExperimentDir, game: str, generations: int) -> subprocess.Popen:
    """Launch the real rollout->train->eval loop for a config that survived
    the quick screen. Raises Busy if another evolve is already running."""
    if not _LOCKS["evolve"].acquire(blocking=False):
        raise Busy("an evolve run is already in progress")
    try:
        if _busy("evolve"):
            raise Busy("an evolve run is already in progress")
        cmd = [
            sys.executable, "-m", "slm_rl.cli", "evolve",
            "--game", game,
            "--generations", str(generations),
            "--run-id", exp.run_id,
            "--config-dir", str(exp.config_dir),
            "--warm-start",
        ]
        log = exp.log_path("evolve")
        proc = _spawn(cmd, log)
        _mark_active("evolve", proc)
        return proc
    finally:
        _LOCKS["evolve"].release()


def _spawn(cmd: list[str], log_path: Path) -> subprocess.Popen:
    log_file = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)


def tail_log(exp: ExperimentDir, kind: str, lines: int = 50) -> list[str]:
    """Last `lines` lines of the subprocess log. Read-only."""
    path = exp.log_path(kind)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return f.readlines()[-lines:]
