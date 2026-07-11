"""Pure, read-only summarization of an experiment's rollout JSONL into
scoreboard numbers. Reads only — never mutates `runs/` (CODING_GUIDELINE
invariant 5). No caching: called fresh on every scoreboard poll, over files
that are at most a few thousand short lines (quick experiments cap at 200
episodes), so re-reading is cheap and avoids a second source of truth to
keep in sync with the JSONL on disk.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _iter_records(run_dir: Path):
    """Yield every JSONL record across every generation's rollouts/*.jsonl
    under `run_dir` (rollout JSONL layout: one file can hold ALL episodes,
    interleaved by write order -- see plan 013 Current state). Malformed
    trailing lines (a write in progress) are skipped, not fatal -- the
    scoreboard polls a live subprocess."""
    for path in sorted(run_dir.glob("generations/gen_*/rollouts/*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def experiment_stats(run_dir: Path | str) -> dict[str, Any]:
    """Group records by `episode_id` and summarize. Returns:
    {episodes, mean_score, median_score, max_score, action_mix (pct by
    parsed_action), intervention_episodes, status}.

    `status` is "no_data" (nothing written yet — the subprocess may still be
    starting), "running" (episodes seen but none has a terminal `outcome`
    line yet, or fewer records than a healthy episode), or "done"-equivalent
    reporting via the caller (this function itself never inspects whether
    the subprocess is alive; that's process state, not file state).
    """
    run_dir = Path(run_dir)
    episodes: dict[str, dict[str, Any]] = defaultdict(lambda: {"score": None, "flagged": False})
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

    n = len(episodes)
    scores = sorted(e["score"] for e in episodes.values() if e["score"] is not None)
    intervention_episodes = sum(1 for e in episodes.values() if e["flagged"])

    action_mix = {}
    if total_decisions:
        action_mix = {
            action: round(100.0 * count / total_decisions, 1)
            for action, count in action_counts.most_common()
        }

    if n == 0:
        status = "no_data"
    elif len(scores) < n:
        status = "running"
    else:
        status = "complete"

    return {
        "episodes": n,
        "mean_score": round(sum(scores) / len(scores), 2) if scores else None,
        "median_score": _median(scores),
        "max_score": max(scores) if scores else None,
        "action_mix": action_mix,
        "intervention_episodes": intervention_episodes,
        "status": status,
    }


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


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    n = len(values)
    mid = n // 2
    if n % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0
