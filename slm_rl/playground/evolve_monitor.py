"""Discover evolve / SFT runs and expose metrics/logs for the Evolve UI.

Covers CLI runs under ``<home>/<run_id>/`` and playground experiment evolve
logs. Parses live stdout for phase + TRL loss when ``train.metrics.jsonl``
is not yet written.
"""

from __future__ import annotations

import json
import os
import re
import signal
from pathlib import Path
from typing import Any

_RUN_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,80}$")
_SKIP_DIRS = frozenset({"playground", "packs", "teachers", "logs", ".cache"})

_PHASE_BASELINE = re.compile(r"\[evolve\] gen (\d+) baseline")
_PHASE_START = re.compile(r"\[evolve\] gen (\d+): start \(")
_PHASE_ROLLOUT = re.compile(
    r"\[evolve\] gen (\d+): (?:model rollout|teacher rollout|loading champion for rollout|using baked pack)"
)
_PHASE_ROLLOUT_DONE = re.compile(r"\[evolve\] gen (\d+): rollout done")
_PHASE_TRAIN = re.compile(r"\[evolve\] gen (\d+): train start")
_PHASE_TRAIN_DONE = re.compile(r"\[evolve\] gen (\d+): train done")
# Labels are either "gen N: …" or "gen N gate eval: …" (space, no colon).
_PHASE_EVAL = re.compile(
    r"\[evolve\] gen (\d+)(?::|\s).*(?:eval|gate skip)",
)
_PHASE_PROMOTE = re.compile(r"\[evolve\] gen (\d+): PROMOTED")
_PHASE_REJECT = re.compile(r"\[evolve\] gen (\d+): rejected")
_TRL_LOSS = re.compile(
    r"['\"]loss['\"]\s*:\s*(?P<loss>[-+eE0-9.]+)"
    r"(?:.*?['\"]epoch['\"]\s*:\s*(?P<epoch>[-+eE0-9.]+))?"
)
_PRIMARY = re.compile(r"primary=(?P<primary>[-+eE0-9.]+)")
_LOG_RUN_ID = re.compile(r"\[evolve\]\s+run_id=(?P<run_id>[a-zA-Z0-9._-]+)")
# Banner: `[evolve] run_id=… generations=5 …` and
# `[evolve] next generation=1 (will run until 5)`.
_TARGET_GENS = re.compile(r"\[evolve\].*\bgenerations=(?P<n>\d+)")
_RANGE_GENS = re.compile(
    r"\[evolve\].*?(?:next generation=|next=)(?P<start>\d+)\s+\(will run until\s+(?P<end>\d+)\)"
)
_EARLY_STOP = re.compile(r"\[evolve\] early stop at gen (?P<g>\d+)")
# TRL progress / metrics: ` 75%|…| 6/8 [02:57<01:00, 30.20s/it]` and
# `{'loss': …, 'kl': 1.41, 'entropy': 2.3, 'reward': -0.76, …}`.
_TRL_PROGRESS = re.compile(r"\|[^\n]*?\|\s*(?P<cur>\d+)/(?P<total>\d+)\s*\[")
_TRL_KL = re.compile(r"['\"]kl['\"]\s*:\s*(?P<v>[-+eE0-9.]+)")
# Rich traceback last line: `KeyError: 'intervention_rate'` (box chars stripped).
_CRASH_EXC = re.compile(r"^[\s│]*([A-Za-z_][\w.]*Error|[A-Za-z_][\w.]*Exception):\s*(.+)$")
_TRL_ENTROPY = re.compile(r"['\"]entropy['\"]\s*:\s*(?P<v>[-+eE0-9.]+)")
_TRL_REWARD = re.compile(r"['\"]reward['\"]\s*:\s*(?P<v>[-+eE0-9.]+)")


def log_dirs(home: Path | str) -> list[Path]:
    home = Path(home)
    return [home.parent / "logs", home / "logs"]


def find_log_path(home: Path | str, run_id: str) -> Path | None:
    candidates: list[Path] = []
    for d in log_dirs(home):
        candidates.append(d / f"evolve-{run_id}.log")
        candidates.append(d / f"evolve-{run_id}.txt")
    # Playground experiment: run_id pg-<name> → playground/<name>/evolve.log
    if run_id.startswith("pg-"):
        name = run_id[3:]
        candidates.append(Path(home) / "playground" / name / "evolve.log")
        candidates.append(Path(home) / "playground" / name / "logs" / "evolve.log")
    for p in candidates:
        if p.is_file():
            return p
    # Prefix / short-name logs (e.g. evolve-boxing-sft.log for boxing-sft-001)
    for d in log_dirs(home):
        if not d.is_dir():
            continue
        for p in sorted(d.glob("evolve-*.log")):
            stem = p.name.removeprefix("evolve-").removesuffix(".log")
            if run_id == stem or run_id.startswith(stem + "-") or stem.startswith(run_id):
                # Confirm content when ambiguous.
                head = tail_file(p, max_bytes=4000)
                m = _LOG_RUN_ID.search(head)
                if m is None or m.group("run_id") == run_id:
                    return p
    return None


def find_pid_path(home: Path | str, run_id: str) -> Path | None:
    for d in log_dirs(home):
        p = d / f"evolve-{run_id}.pid"
        if p.is_file():
            return p
    lpath = find_log_path(home, run_id)
    if lpath is not None:
        sibling = lpath.with_suffix(".pid")
        if sibling.is_file():
            return sibling
        # evolve-foo.log → evolve-foo.pid
        alt = lpath.parent / (lpath.stem + ".pid")
        if alt.is_file():
            return alt
    return None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_pid(path: Path | None) -> int | None:
    if path is None or not path.is_file():
        return None
    try:
        pid = int(path.read_text(encoding="utf-8").strip().split()[0])
    except (ValueError, OSError, IndexError):
        return None
    return pid if _pid_alive(pid) else None


def tail_file(path: Path, max_bytes: int = 256_000) -> str:
    if not path.is_file():
        return ""
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
        data = f.read()
    text = data.decode("utf-8", errors="replace")
    if size > max_bytes:
        text = f"…(truncated; full log at {path})\n" + text
    return text


def discover_run_ids(home: Path | str) -> list[str]:
    home = Path(home)
    found: set[str] = set()
    if home.is_dir():
        for child in home.iterdir():
            if not child.is_dir() or child.name.startswith(".") or child.name in _SKIP_DIRS:
                continue
            if (child / "run_config.yaml").is_file() or (child / "registry.json").is_file():
                if _RUN_ID_RE.match(child.name):
                    found.add(child.name)
        pg = home / "playground"
        if pg.is_dir():
            for child in pg.iterdir():
                if child.is_dir() and (child / "run_config.yaml").is_file():
                    rid = f"pg-{child.name}"
                    if _RUN_ID_RE.match(rid):
                        found.add(rid)
    for d in log_dirs(home):
        if not d.is_dir():
            continue
        for p in d.glob("evolve-*.log"):
            rid = p.name.removeprefix("evolve-").removesuffix(".log")
            head = tail_file(p, max_bytes=4000)
            m = _LOG_RUN_ID.search(head)
            if m and _RUN_ID_RE.match(m.group("run_id")):
                found.add(m.group("run_id"))
            elif _RUN_ID_RE.match(rid):
                found.add(rid)
        for p in d.glob("evolve-*.pid"):
            rid = p.name.removeprefix("evolve-").removesuffix(".pid")
            # Prefer matching log's embedded run_id over short pid stem.
            log = p.with_suffix(".log")
            if log.is_file():
                m = _LOG_RUN_ID.search(tail_file(log, max_bytes=4000))
                if m and _RUN_ID_RE.match(m.group("run_id")):
                    found.add(m.group("run_id"))
                    continue
            if _RUN_ID_RE.match(rid):
                found.add(rid)
    return sorted(found)


def run_dir(home: Path | str, run_id: str) -> Path | None:
    """Directory that contains ``generations/``, ``registry.json``, etc."""
    home = Path(home)
    if run_id.startswith("pg-"):
        # Playground: <home>/playground/<name>/pg-<name>/
        name = run_id[3:]
        nested = home / "playground" / name / run_id
        if nested.is_dir():
            return nested
        flat = home / "playground" / name
        return flat if flat.is_dir() else None
    p = home / run_id
    return p if p.is_dir() else None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _parse_phase(log_text: str) -> dict[str, Any]:
    # ponytail: empty log = never started — not "starting" (UI treated that as dead).
    if not (log_text or "").strip():
        return {"phase": "", "phase_generation": None}
    phase = "starting"
    gen = 0
    for line in log_text.splitlines():
        if m := _PHASE_BASELINE.search(line):
            gen = int(m.group(1))
            phase = "baseline"
        elif m := _PHASE_START.search(line):
            gen = int(m.group(1))
            phase = "rollout"
        elif m := _PHASE_ROLLOUT.search(line):
            gen = int(m.group(1))
            phase = "rollout"
        elif m := _PHASE_ROLLOUT_DONE.search(line):
            gen = int(m.group(1))
            phase = "rollout_done"
        elif m := _PHASE_TRAIN.search(line):
            gen = int(m.group(1))
            phase = "train"
        elif m := _PHASE_TRAIN_DONE.search(line):
            gen = int(m.group(1))
            phase = "train_done"
        elif m := _PHASE_EVAL.search(line):
            gen = int(m.group(1))
            phase = "eval"
        elif m := _PHASE_PROMOTE.search(line):
            gen = int(m.group(1))
            phase = "promoted"
        elif m := _PHASE_REJECT.search(line):
            gen = int(m.group(1))
            phase = "rejected"
        elif m := _EARLY_STOP.search(line):
            gen = int(m.group("g"))
            phase = "early_stop"
    return {"phase": phase, "phase_generation": gen}


def parse_run_plan(log_text: str) -> dict[str, Any]:
    """Target generation range from the evolve banner lines (no new writes)."""
    target = start = end = None
    early_end = None
    for line in log_text.splitlines():
        if m := _TARGET_GENS.search(line):
            target = int(m.group("n"))
        if m := _RANGE_GENS.search(line):
            start = int(m.group("start"))
            end = int(m.group("end"))
        if m := _EARLY_STOP.search(line):
            early_end = int(m.group("g"))
    if early_end is not None:
        end = early_end
    if start is not None and end is not None:
        target = max(0, end - start + 1)
    return {
        "target_generations": target,
        "start_generation": start,
        "end_generation": end,
    }


def parse_train_progress(log_text: str) -> dict[str, Any]:
    """Latest TRL step progress + kl/entropy/reward from stdout.

    Resets on each ``train start`` so a prior gen's 8/8 doesn't leak into the
    next. Progress lines alone (no metrics dict yet) still update the step.
    """
    step = total = None
    kl = entropy = reward = None
    for line in log_text.splitlines():
        if "train start" in line and "[evolve]" in line:
            step = total = None
            kl = entropy = reward = None
            continue
        if "train done" in line or "[evolve]" in line:
            continue
        # HF weight load uses the same tqdm shape as TRL — skip it.
        if "Loading weights" in line or "Fetching " in line:
            continue
        if m := _TRL_PROGRESS.search(line):
            step = int(m.group("cur"))
            total = int(m.group("total"))
        if m := _TRL_KL.search(line):
            kl = float(m.group("v"))
        if m := _TRL_ENTROPY.search(line):
            entropy = float(m.group("v"))
        if m := _TRL_REWARD.search(line):
            reward = float(m.group("v"))
    return {
        "train_step": step,
        "train_total_steps": total,
        "train_kl": kl,
        "train_entropy": entropy,
        "train_reward": reward,
    }


def parse_train_loss_from_log(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    step = 0
    for line in text.splitlines():
        # Skip evolve summary lines that embed a final loss in metrics={...}.
        if "train done" in line or "[evolve]" in line:
            continue
        m = _TRL_LOSS.search(line)
        if not m:
            continue
        step += 5  # logging_steps default
        row: dict[str, Any] = {
            "split": "train",
            "step": step,
            "loss": float(m.group("loss")),
        }
        if m.group("epoch") is not None:
            row["epoch"] = float(m.group("epoch"))
        rows.append(row)
    return rows


def parse_crash_error(log_text: str) -> str | None:
    """Last exception line from a traceback in evolve.log, or None."""
    idx = log_text.rfind("Traceback")
    if idx < 0:
        return None
    last: str | None = None
    for line in log_text[idx:].splitlines():
        m = _CRASH_EXC.match(line)
        if m:
            last = f"{m.group(1)}: {m.group(2).strip()}"
    return last


def _generation_summaries(root: Path) -> list[dict[str, Any]]:
    gens_dir = root / "generations"
    if not gens_dir.is_dir():
        return []

    out: list[dict[str, Any]] = []
    for gdir in sorted(gens_dir.glob("gen_*")):
        try:
            gen = int(gdir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        metrics = _read_json(gdir / "metrics.json") or {}
        eval_m = metrics.get("eval") or _read_json(gdir / "eval" / "results.json") or {}
        train_m = metrics.get("train") or {}
        gate = metrics.get("gate") or {}
        out.append({
            "generation": gen,
            "primary": eval_m.get("primary"),
            "num_pairs": train_m.get("num_pairs"),
            "loss": train_m.get("loss"),
            "promoted": gate.get("promoted"),
            "gate_reason": gate.get("reason"),
            "has_adapter": (gdir / "adapter").is_dir(),
            "has_train_metrics": (gdir / "train.metrics.jsonl").is_file(),
        })
    return out


def list_evolve_jobs(home: Path | str) -> list[dict[str, Any]]:
    home = Path(home)
    jobs: list[dict[str, Any]] = []
    for run_id in discover_run_ids(home):
        root = run_dir(home, run_id)
        lpath = find_log_path(home, run_id)
        pid = _read_pid(find_pid_path(home, run_id))
        log_text = tail_file(lpath, max_bytes=120_000) if lpath else ""
        phase = _parse_phase(log_text)
        plan = parse_run_plan(log_text)
        train_prog = parse_train_progress(log_text)
        gens = _generation_summaries(root) if root else []
        registry = _read_json(root / "registry.json") if root else None
        cfg = None
        cfg_path = None
        if root:
            for candidate in (
                root / "run_config.yaml",
                root.parent / "run_config.yaml",
                root.parent / "config" / "default.yaml",
            ):
                if candidate.is_file():
                    cfg_path = candidate
                    break
        if cfg_path is not None:
            try:
                import yaml

                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                cfg = None
        last_primary = None
        for g in reversed(gens):
            if g.get("primary") is not None:
                last_primary = g["primary"]
                break
        # Fallback: scrape log
        if last_primary is None:
            for m in _PRIMARY.finditer(log_text):
                last_primary = float(m.group("primary"))
        started_at = None
        if lpath is not None and lpath.is_file():
            try:
                started_at = int(lpath.stat().st_ctime)
            except OSError:
                started_at = None
        jobs.append({
            "run_id": run_id,
            "running": pid is not None,
            "pid": pid,
            "path": str(root) if root else None,
            "log_path": str(lpath) if lpath else None,
            "game": (cfg or {}).get("game") if isinstance(cfg, dict) else None,
            "model": (cfg or {}).get("model") if isinstance(cfg, dict) else None,
            "backend": (cfg or {}).get("backend") if isinstance(cfg, dict) else None,
            "champion": (registry or {}).get("champion") if isinstance(registry, dict) else None,
            "generations": gens,
            "last_primary": last_primary,
            "started_at": started_at,
            **phase,
            **plan,
            **train_prog,
        })
    jobs.sort(key=lambda j: (not j["running"], j["run_id"]))
    return jobs


def read_metrics_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _process_running(home: Path | str, run_id: str) -> bool:
    """True when the evolve OS process (or playground job slot) is alive.

    Log phase alone is not enough: after Docker/API restarts the last line
    can still say ``rollout`` while nothing is collecting episodes.
    """
    if _read_pid(find_pid_path(home, run_id)) is not None:
        return True
    # Playground: in-memory job table (written at launch) — covers the window
    # before/without a readable pid file.
    if run_id.startswith("pg-"):
        try:
            from slm_rl.playground.experiments import active_jobs_for

            if "evolve" in active_jobs_for(run_id[3:]):
                return True
        except Exception:  # noqa: BLE001 — monitor must stay read-only-safe
            pass
    return False


def job_metrics(home: Path | str, run_id: str) -> dict[str, Any]:
    if not _RUN_ID_RE.match(run_id):
        raise ValueError(f"invalid run_id: {run_id!r}")
    home = Path(home)
    root = run_dir(home, run_id)
    lpath = find_log_path(home, run_id)
    log_text = tail_file(lpath, max_bytes=2_000_000) if lpath else ""

    train_rows: list[dict[str, Any]] = []
    if root and (root / "generations").is_dir():
        for gdir in sorted((root / "generations").glob("gen_*")):
            train_rows.extend(read_metrics_jsonl(gdir / "train.metrics.jsonl"))

    if not any(r.get("split") == "train" and "loss" in r for r in train_rows):
        train_rows = train_rows + parse_train_loss_from_log(log_text)

    gens = _generation_summaries(root) if root else []
    eval_curve = [
        {"generation": g["generation"], "primary": g["primary"], "split": "eval"}
        for g in gens
        if g.get("primary") is not None
    ]
    phase = _parse_phase(log_text)
    plan = parse_run_plan(log_text)
    train_prog = parse_train_progress(log_text)
    pid = _read_pid(find_pid_path(home, run_id))
    running = _process_running(home, run_id)
    registry = _read_json(root / "registry.json") if root else None
    champion = (registry or {}).get("champion") if isinstance(registry, dict) else None
    # ponytail: log ctime ≈ evolve start; good enough for a workshop ETA.
    started_at = None
    if lpath is not None and lpath.is_file():
        try:
            started_at = int(lpath.stat().st_ctime)
        except OSError:
            started_at = None
    crash = None if running else parse_crash_error(log_text)
    return {
        "run_id": run_id,
        "log_path": str(lpath) if lpath else None,
        "path": str(root) if root else None,
        "running": running,
        "pid": pid,
        "champion": champion,
        "started_at": started_at,
        "crash_error": crash,
        "train": [r for r in train_rows if r.get("split") == "train"],
        "eval": eval_curve,
        "generations": gens,
        "points": train_rows,
        **phase,
        **plan,
        **train_prog,
    }


def job_log(home: Path | str, run_id: str, max_bytes: int = 256_000) -> str:
    if not _RUN_ID_RE.match(run_id):
        raise ValueError(f"invalid run_id: {run_id!r}")
    lpath = find_log_path(home, run_id)
    if lpath is None:
        return ""
    return tail_file(lpath, max_bytes=max_bytes)


def stop_evolve_job(home: Path | str, run_id: str) -> dict[str, Any]:
    if not _RUN_ID_RE.match(run_id):
        raise ValueError(f"invalid run_id: {run_id!r}")
    pid = _read_pid(find_pid_path(home, run_id))
    if pid is None:
        raise FileNotFoundError(f"no running evolve job for {run_id!r}")
    os.kill(pid, signal.SIGTERM)
    return {"ok": True, "run_id": run_id, "pid": pid}
