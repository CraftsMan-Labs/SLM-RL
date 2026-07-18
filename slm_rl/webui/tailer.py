"""Tail `runs/<run_id>/generations/gen_*/rollouts/*.jsonl` and reduce each
`RolloutRecord` to a small wire payload for the live-play viewer.

Read-only observer (CODING_GUIDELINE invariant 5): opens files read-only,
never writes into `runs/`. Stdlib only.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _sorted_rollout_files(
    run_dir: Path, generation: int | None = None
) -> list[Path]:
    """Rollout JSONLs in (generation, filename) order. Re-globbed on every
    call so new gen_NNN dirs/files created after the tailer starts are
    picked up — runs are live-appended and resumed runs add generations.
    When `generation` is set, only that gen's rollouts are opened (frame
    replay must not walk GB-scale sibling gens)."""
    if generation is not None:
        gen_dir = run_dir / "generations" / f"gen_{generation:03d}"
        if not gen_dir.is_dir():
            return []
        return sorted((gen_dir / "rollouts").glob("*.jsonl"))
    gen_dirs = sorted(run_dir.glob("generations/gen_*"), key=lambda p: p.name)
    files: list[Path] = []
    for gen_dir in gen_dirs:
        files.extend(sorted((gen_dir / "rollouts").glob("*.jsonl")))
    return files


def iter_run_records(
    run_dir: Path,
    poll_interval: float = 0.5,
    stop: threading.Event | None = None,
    *,
    generation: int | None = None,
    episode_id: str | None = None,
    follow: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield parsed record dicts from a run's rollout JSONLs, oldest first.

    Catches up on existing lines first, then polls for appends and new
    files/directories. Tolerates a truncated trailing line (the writer may
    be mid-line): the byte offset is only advanced past a line that parsed
    cleanly, so a partial line is retried on the next poll. Never raises for
    bad data. Exits when `stop` is set (or immediately, if already set).
    When `follow` is False, returns after one catch-up pass (no live poll).

    Optional filters (frame replay):
    - `generation`: only open that gen's JSONLs
    - `episode_id`: skip `json.loads` unless the line contains the id
      (Atari warm-starts are GB-scale; parsing every line freezes Watch screen)
    """
    offsets: dict[Path, int] = {}

    def _drain(path: Path) -> Iterator[dict[str, Any]]:
        offset = offsets.get(path, 0)
        with path.open("r", encoding="utf-8") as f:
            f.seek(offset)
            while True:
                line = f.readline()
                if not line:
                    break
                if not line.endswith("\n"):
                    # Partial trailing line — writer may still be flushing it.
                    # Don't advance the offset; retry from here next poll.
                    break
                stripped = line.strip()
                if not stripped:
                    offset = f.tell()
                    continue
                # Advance past non-matching lines without parsing (huge files).
                if episode_id is not None and episode_id not in stripped:
                    offset = f.tell()
                    continue
                try:
                    rec = json.loads(stripped)
                except json.JSONDecodeError:
                    # Malformed/partial line despite the trailing newline
                    # (e.g. writer crashed mid-flush) — retry next poll.
                    break
                offset = f.tell()
                if episode_id is not None and rec.get("episode_id") != episode_id:
                    continue
                if generation is not None:
                    g = rec.get("generation")
                    # None: old-schema rows in a gen-scoped file still count.
                    if g is not None and g != generation:
                        continue
                yield rec
        offsets[path] = offset

    while True:
        for path in _sorted_rollout_files(run_dir, generation=generation):
            yield from _drain(path)
        if stop is not None and stop.is_set():
            return
        if not follow:
            return
        time.sleep(poll_interval)
        if stop is not None and stop.is_set():
            return


def to_event(rec: dict[str, Any]) -> dict[str, Any]:
    """Reduce a raw record dict to the wire payload sent to the browser.

    Drops `prompt_messages` (heavy, full chat history) but keeps `observed`
    — the content of the last user message, i.e. what the model saw this
    turn. Uses `.get(...)` throughout: old-schema files may lack fields
    (CODING_GUIDELINE invariant on resume-tolerance).
    """
    observed = ""
    for msg in reversed(rec.get("prompt_messages", []) or []):
        if msg.get("role") == "user":
            observed = msg.get("content", "")
            break
    return {
        "episode_id": rec.get("episode_id"),
        "generation": rec.get("generation"),
        "step_idx": rec.get("step_idx"),
        "parsed_action": rec.get("parsed_action"),
        "completion": rec.get("completion"),
        "parse_status": rec.get("parse_status"),
        "reward": rec.get("reward"),
        "cum_reward": rec.get("cum_reward"),
        "terminated": rec.get("terminated"),
        "truncated": rec.get("truncated"),
        "outcome": rec.get("outcome"),
        "monitor_flags": rec.get("monitor_flags"),
        "model_id": rec.get("model_id"),
        "seed": rec.get("seed"),
        "observed": observed,
    }
