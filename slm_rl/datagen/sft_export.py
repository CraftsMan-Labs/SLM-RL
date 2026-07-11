"""Trajectory selection + export to SFT chat datasets for the reject_sft
strategy: keep winning / top-quantile, monitor-clean trajectories with a
diversity quota (see docs/DECISIONS.md D10).

`select_episodes` is factored out so the (future) GRPO dataset builder reuses
the identical filter.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from slm_rl.config.schema import TrainConfig

_NESTED = ("prompt_messages", "monitor_flags", "legal_actions")


def _iter_records(dataset_path: Path) -> Iterator[dict]:
    """Yield record dicts from a parquet file, a .jsonl file, or a dir of
    .jsonl files. Nested fields stored as JSON strings (parquet) are decoded."""
    path = Path(dataset_path)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for f in files:
        if f.suffix == ".parquet":
            import pyarrow.parquet as pq

            for row in pq.read_table(f).to_pylist():
                for key in _NESTED:
                    if isinstance(row.get(key), str):
                        row[key] = json.loads(row[key])
                yield row
        else:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield json.loads(line)


def select_episodes(dataset_path: Path, cfg: TrainConfig) -> list[list[dict]]:
    """Group records into episodes and select the ones worth training on.
    Returns a list of episodes, each a step-ordered list of records."""
    episodes: dict[str, list[dict]] = defaultdict(list)
    for rec in _iter_records(dataset_path):
        episodes[rec["episode_id"]].append(rec)
    for steps in episodes.values():
        steps.sort(key=lambda r: r["step_idx"])

    def dirty(steps: list[dict]) -> bool:
        return any(
            s["monitor_flags"].get("intervention") or s["monitor_flags"].get("truncated_by_monitor")
            for s in steps
        )

    def final_return(steps: list[dict]) -> float:
        return steps[-1]["cum_reward"]

    def outcome(steps: list[dict]) -> str | None:
        return steps[-1].get("outcome")

    candidates = list(episodes.values())
    if cfg.exclude_monitor_flagged:
        candidates = [s for s in candidates if not dirty(s)] or candidates

    returns = sorted(final_return(s) for s in candidates)
    # top `selection_quantile` fraction by return
    cutoff_idx = int(len(returns) * (1 - cfg.selection_quantile))
    cutoff = returns[min(cutoff_idx, len(returns) - 1)] if returns else 0.0

    wins = [s for s in candidates if outcome(s) == "win"]
    top = [s for s in candidates if final_return(s) >= cutoff]
    selected = {id(s): s for s in wins + top}.values()  # union, dedup by identity
    selected = list(selected) or top  # fallback keeps the loop fed (R7)

    # win_turn_cap filters the FINAL union: long wins tend to have the
    # highest cumulative return (per-turn format reward), so filtering only
    # the wins list would let them re-enter via `top` almost every time.
    # The cap never drops losses, so it cannot empty a non-empty selection
    # (the fallback below re-adds a win whenever any was selected).
    if cfg.win_turn_cap > 0:
        surviving = [s for s in selected if not (outcome(s) == "win" and len(s) > cfg.win_turn_cap)]
        # ponytail: if the cap removed every win, keep the shortest win so a
        # winning demonstration always survives.
        if wins and not any(outcome(s) == "win" for s in surviving):
            surviving.append(min(wins, key=len))
        selected = surviving

    # diversity quota: cap identical action sequences, prefer higher return
    by_seq: dict[tuple, list[list[dict]]] = defaultdict(list)
    for steps in selected:
        seq = tuple(s["parsed_action"] for s in steps)
        by_seq[seq].append(steps)
    result: list[list[dict]] = []
    for seq_episodes in by_seq.values():
        seq_episodes.sort(key=final_return, reverse=True)
        result.extend(seq_episodes[: cfg.max_duplicate_action_sequences])
    return result


def export_sft_dataset(dataset_path: Path, out_path: Path, cfg: TrainConfig) -> int:
    """Write TRL conversational prompt-completion pairs; return pair count."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for steps in select_episodes(dataset_path, cfg):
            is_win = steps[-1].get("outcome") == "win"
            for rec in steps:
                if rec["parse_status"] == "fallback_random":
                    continue  # completion wouldn't match the action taken
                prompt = rec["prompt_messages"][:2]  # clean system+user (drop retry turns)
                if not prompt:
                    continue
                # ponytail: teacher completions carry a verbalized rationale
                # (plan 002) and are trusted verbatim; model-generated
                # completions can contain retry junk, so those are still
                # rebuilt from the canonical parsed_action.
                is_teacher = str(rec.get("model_id", "")).startswith("teacher:")
                content = rec["completion"] if is_teacher else f"ACTION: {rec['parsed_action']}"
                row = {
                    "prompt": prompt,
                    "completion": [{"role": "assistant", "content": content}],
                }
                # ponytail: duplication ≈ sample weighting; real per-sample
                # weights if the trainer ever supports them. The final
                # decision pair of a win (discount γ^0, highest-signal
                # demonstration) is written sft_win_final_dup times.
                is_final = is_win and rec is steps[-1]
                reps = cfg.sft_win_final_dup if is_final else 1
                for _ in range(reps):
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    pairs += 1
    return pairs
