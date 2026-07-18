"""Pure, read-only summarization of an experiment's rollout JSONL into
scoreboard numbers. Reads only — never mutates `runs/` (CODING_GUIDELINE
invariant 5).

Workshop quick-runs are tiny; Atari warm-starts can be GB-scale JSONL. A
full re-parse on every `/api/experiments` poll (every few seconds) freezes
the playground (~100s measured on a 1.3GB demon-attack file). We therefore:
  1. cache by (path, mtime_ns, size) fingerprint
  2. for oversized files, scan only a trailing byte window (recent episodes)
"""

from __future__ import annotations

from statistics import median

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

# ponytail: 4MiB tail keeps scoreboard snappy; full scan if you need exact
# historical mixes on multi-GB Atari rolls — use metrics.json after a gen.
_MAX_BYTES_PER_FILE = 4 * 1024 * 1024
_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_CACHE_MAX = 128


def _rollout_paths(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("generations/gen_*/rollouts/*.jsonl"))


def _fingerprint(run_dir: Path) -> tuple[Any, ...]:
    """Invalidate cache when any rollout file grows or is rewritten."""
    parts: list[Any] = [str(run_dir)]
    for path in _rollout_paths(run_dir):
        try:
            st = path.stat()
        except OSError:
            continue
        parts.append((str(path), st.st_mtime_ns, st.st_size))
    return tuple(parts)


def _iter_file_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from one JSONL. Oversized files: last N bytes only."""
    try:
        size = path.stat().st_size
    except OSError:
        return
    if size == 0:
        return
    with path.open("rb") as f:
        if size > _MAX_BYTES_PER_FILE:
            f.seek(size - _MAX_BYTES_PER_FILE)
            # drop partial first line after seek
            f.readline()
        text = f.read().decode("utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _iter_records(run_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield JSONL records under `run_dir` (capped per file — see module doc)."""
    for path in _rollout_paths(run_dir):
        yield from _iter_file_records(path)


def _summarize(run_dir: Path) -> dict[str, Any]:
    episodes: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"score": None, "flagged": False, "outcome": None}
    )
    action_counts: Counter[str] = Counter()
    total_decisions = 0

    for rec in _iter_records(run_dir):
        ep_id = rec.get("episode_id")
        if ep_id is None:
            continue
        entry = episodes[ep_id]

        action = rec.get("parsed_action")
        if action is not None:
            action_counts[action] += 1
            total_decisions += 1

        if rec.get("monitor_flags"):
            entry["flagged"] = True

        outcome = rec.get("outcome")
        if outcome is not None:
            entry["score"] = _parse_score(outcome)
            entry["outcome"] = outcome

    n = len(episodes)
    scores = sorted(e["score"] for e in episodes.values() if e["score"] is not None)
    intervention_episodes = sum(1 for e in episodes.values() if e["flagged"])
    terminal = [e["outcome"] for e in episodes.values() if e["outcome"] is not None]
    win_rate = (
        round(sum(1 for o in terminal if o == "win") / len(terminal), 3) if terminal else None
    )

    action_mix = {}
    if total_decisions:
        action_mix = {
            action: round(100.0 * count / total_decisions, 1)
            for action, count in action_counts.most_common()
        }

    # Status still keys off numeric scores (Atari `score:<n>`); board-game
    # win/loss terminals contribute to win_rate but not mean/median/max.
    if n == 0:
        status = "no_data"
    elif len(scores) < n:
        status = "running"
    else:
        status = "complete"

    return {
        "episodes": n,
        "mean_score": round(sum(scores) / len(scores), 2) if scores else None,
        "median_score": median(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "action_mix": action_mix,
        "intervention_episodes": intervention_episodes,
        "status": status,
        "win_rate": win_rate,
    }


def experiment_stats(run_dir: Path | str) -> dict[str, Any]:
    """Group records by `episode_id` and summarize. Returns:
    {episodes, mean_score, median_score, max_score, action_mix (pct by
    parsed_action), intervention_episodes, status, win_rate}.

    `status` is "no_data" (nothing written yet — the subprocess may still be
    starting), "running" (episodes seen but none has a terminal `outcome`
    line yet, or fewer records than a healthy episode), or "complete".
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return _summarize(run_dir)

    key = _fingerprint(run_dir)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    stats = _summarize(run_dir)
    if len(_CACHE) >= _CACHE_MAX:
        # ponytail: drop an arbitrary old entry; fingerprints churn with mtime
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = stats
    return stats


def exhibition_scores(theater_dir: Path | str) -> dict[str, Any]:
    """Plan 020 A/B score strip: `experiment_stats` (incl. win_rate), one
    entry per side present under `theater_dir`.
    Read-only, same caching/cap stance as `experiment_stats`.

    Also surfaces `theater/status.json` (when present) as `run` so the UI can
    show "champion waits for base (3/10)" / failure instead of a silent wait.
    """
    import json

    theater_dir = Path(theater_dir)
    result: dict[str, Any] = {}
    for side in ("base", "champion"):
        side_dir = theater_dir / side
        if not side_dir.exists():
            continue
        result[side] = experiment_stats(side_dir)
    status_path = theater_dir / "status.json"
    if status_path.is_file():
        try:
            result["run"] = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return result


def _parse_score(outcome: str) -> float | None:
    """`outcome` is "win" | "loss" | "draw" | "score:<n>" (Atari). Only the
    Atari form carries a numeric score; board-game outcomes contribute to
    episode counts but not to mean/median/max score."""
    if isinstance(outcome, str) and outcome.startswith("score:"):
        try:
            return float(outcome.split(":", 1)[1])
        except ValueError:
            return None
    return None
