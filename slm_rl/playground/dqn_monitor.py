"""Discover train-dqn jobs and expose metrics/logs for the Teachers UI.

Read-only observer over:
  <home>/teachers/dqn-<game>.pt
  <home>/teachers/dqn-<game>.metrics.jsonl
  <repo>/logs/train-dqn-<game>.log   (when home is <repo>/runs)
"""

from __future__ import annotations

import json
import os
import re
import signal
from pathlib import Path
from typing import Any

_TRAIN_LINE = re.compile(
    r"decisions=(?P<decisions>\d+)\s+"
    r"episodes=(?P<episodes>\d+)\s+"
    r"eps=(?P<eps>[-+eE0-9.]+)\s+"
    r"mean_ep_reward_last20=(?P<mean_ep_reward>[-+eE0-9.]+)\s+"
    r"loss=(?P<loss>\S+)"
)
_EVAL_LINE = re.compile(
    r"eval\s+decisions=(?P<decisions>\d+)\s+"
    r"episodes=(?P<episodes>\d+)\s+"
    r"mean_ep_reward=(?P<mean_ep_reward>[-+eE0-9.]+)"
)
_GAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}$")


def teachers_dir(home: Path | str) -> Path:
    return Path(home) / "teachers"


def log_dirs(home: Path | str) -> list[Path]:
    home = Path(home)
    return [
        home / "teachers" / "logs",
        home.parent / "logs",
        home / "logs",
    ]


def metrics_path(home: Path | str, game: str) -> Path:
    return teachers_dir(home) / f"dqn-{game}.metrics.jsonl"


def checkpoint_path(home: Path | str, game: str) -> Path:
    return teachers_dir(home) / f"dqn-{game}.pt"


def find_log_path(home: Path | str, game: str) -> Path | None:
    candidates = []
    for d in log_dirs(home):
        candidates.append(d / f"train-dqn-{game}.log")
    # Also accept a sibling of the checkpoint.
    candidates.append(teachers_dir(home) / f"dqn-{game}.log")
    for p in candidates:
        if p.is_file():
            return p
    return None


def find_pid_path(home: Path | str, game: str) -> Path | None:
    for d in log_dirs(home):
        p = d / f"train-dqn-{game}.pid"
        if p.is_file():
            return p
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


def _checkpoint_meta(path: Path) -> dict[str, Any]:
    """File stats only — no torch import (playground stays light)."""
    if not path.is_file():
        return {}
    st = path.stat()
    return {
        "checkpoint_bytes": st.st_size,
        "checkpoint_mtime": st.st_mtime,
    }


def _latest_progress(home: Path, game: str) -> dict[str, Any]:
    """Last train/eval point from metrics JSONL or text log (no torch)."""
    data = job_metrics(home, game)
    train = data.get("train") or []
    ev = data.get("eval") or []
    out: dict[str, Any] = {}
    if train:
        last = train[-1]
        out["last_decisions"] = last.get("decisions")
        out["last_train_reward"] = last.get("mean_ep_reward")
        out["last_loss"] = last.get("loss")
    if ev:
        out["last_eval_reward"] = ev[-1].get("mean_ep_reward")
    return out


def discover_games(home: Path | str) -> list[str]:
    games: set[str] = set()
    tdir = teachers_dir(home)
    if tdir.is_dir():
        for p in tdir.glob("dqn-*.pt"):
            games.add(p.stem.removeprefix("dqn-"))
        for p in tdir.glob("dqn-*.metrics.jsonl"):
            name = p.name.removeprefix("dqn-").removesuffix(".metrics.jsonl")
            if name:
                games.add(name)
    for d in log_dirs(home):
        if not d.is_dir():
            continue
        for p in d.glob("train-dqn-*.log"):
            games.add(p.name.removeprefix("train-dqn-").removesuffix(".log"))
        for p in d.glob("train-dqn-*.pid"):
            games.add(p.name.removeprefix("train-dqn-").removesuffix(".pid"))
    return sorted(g for g in games if _GAME_RE.match(g))


def list_dqn_jobs(home: Path | str) -> list[dict[str, Any]]:
    home = Path(home)
    jobs: list[dict[str, Any]] = []
    for game in discover_games(home):
        ckpt = checkpoint_path(home, game)
        mpath = metrics_path(home, game)
        lpath = find_log_path(home, game)
        pid_path = find_pid_path(home, game)
        pid = _read_pid(pid_path)
        meta = _checkpoint_meta(ckpt) if ckpt.is_file() else {}
        progress = _latest_progress(home, game)
        jobs.append({
            "game": game,
            "running": pid is not None,
            "pid": pid,
            "checkpoint": str(ckpt) if ckpt.is_file() else None,
            "metrics_path": str(mpath) if mpath.is_file() else None,
            "log_path": str(lpath) if lpath else None,
            **meta,
            **progress,
        })
    # Running jobs first, then by game name.
    jobs.sort(key=lambda j: (not j["running"], j["game"]))
    return jobs


def parse_train_log_metrics(text: str) -> list[dict[str, Any]]:
    """Parse plain train-dqn stdout into train/eval metric rows."""
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        m = _EVAL_LINE.search(line)
        if m:
            rows.append({
                "split": "eval",
                "decisions": int(m.group("decisions")),
                "episodes": int(m.group("episodes")),
                "mean_ep_reward": float(m.group("mean_ep_reward")),
            })
            continue
        m = _TRAIN_LINE.search(line)
        if m:
            loss_raw = m.group("loss")
            loss: float | None
            try:
                loss = None if loss_raw == "n/a" else float(loss_raw)
            except ValueError:
                loss = None
            rows.append({
                "split": "train",
                "decisions": int(m.group("decisions")),
                "episodes": int(m.group("episodes")),
                "eps": float(m.group("eps")),
                "mean_ep_reward": float(m.group("mean_ep_reward")),
                "loss": loss,
            })
    return rows


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


def job_metrics(home: Path | str, game: str) -> dict[str, Any]:
    if not _GAME_RE.match(game):
        raise ValueError(f"invalid game id: {game!r}")
    home = Path(home)
    mpath = metrics_path(home, game)
    rows = read_metrics_jsonl(mpath)
    # Always merge text-log parse so mid-run jobs (pre-metrics) still chart.
    lpath = find_log_path(home, game)
    if lpath is not None:
        text = tail_file(lpath, max_bytes=2_000_000)
        parsed = parse_train_log_metrics(text)
        if parsed:
            # Prefer JSONL when present; else use log. If both, keep JSONL
            # and append only log points beyond the last JSONL decision.
            if rows:
                last = max(
                    (int(r["decisions"]) for r in rows if "decisions" in r),
                    default=0,
                )
                rows = rows + [r for r in parsed if int(r.get("decisions", 0)) > last]
            else:
                rows = parsed
    train = [r for r in rows if r.get("split") == "train"]
    eval_rows = [r for r in rows if r.get("split") == "eval"]
    return {
        "game": game,
        "metrics_path": str(mpath) if mpath.is_file() else None,
        "log_path": str(lpath) if lpath else None,
        "train": train,
        "eval": eval_rows,
        "points": rows,
    }


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


def job_log(home: Path | str, game: str, max_bytes: int = 256_000) -> str:
    if not _GAME_RE.match(game):
        raise ValueError(f"invalid game id: {game!r}")
    lpath = find_log_path(home, game)
    if lpath is None:
        return ""
    return tail_file(lpath, max_bytes=max_bytes)


def stop_dqn_job(home: Path | str, game: str) -> dict[str, Any]:
    """Best-effort SIGTERM of a pid-file job (Teachers UI stop button)."""
    if not _GAME_RE.match(game):
        raise ValueError(f"invalid game id: {game!r}")
    pid_path = find_pid_path(home, game)
    pid = _read_pid(pid_path)
    if pid is None:
        raise FileNotFoundError(f"no running train-dqn job for {game!r}")
    os.kill(pid, signal.SIGTERM)
    return {"ok": True, "game": game, "pid": pid}
