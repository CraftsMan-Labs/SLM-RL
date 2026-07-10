"""Rollout parquet/JSONL -> GRPO prompt dataset.

Unlike sft_export, every decision step of every episode is usable: the reward
is environment-grounded (recomputed from the reconstructed game state), so no
win filter is needed. Each row carries the prompt plus a `game_ctx` JSON
column the reward functions score against.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from slm_rl.config.schema import GameConfig
from slm_rl.datagen.sft_export import _iter_records
from slm_rl.games.mastermind.env import score_guess
from slm_rl.games.registry import get_game

MAX_PROMPTS = 512  # ponytail: fixed cap keeps GRPO generation minutes-long; make it a TrainConfig field if runs need tuning


def export_grpo_dataset(dataset_path: Path, out_path: Path, game_cfg: GameConfig) -> int:
    """Write {"prompt", "game_ctx"} JSONL rows; return row count."""
    if game_cfg.name != "mastermind":
        raise NotImplementedError(
            f"GRPO reward reconstruction only implemented for mastermind, got {game_cfg.name!r}"
        )

    episodes: dict[str, list[dict]] = defaultdict(list)
    for rec in _iter_records(Path(dataset_path)):
        episodes[rec["episode_id"]].append(rec)
    for steps in episodes.values():
        steps.sort(key=lambda r: r["step_idx"])

    game = get_game(game_cfg.name)(game_cfg)
    rows: list[tuple[int, dict]] = []  # (step_idx, row) for later-turn preference
    seen: set[str] = set()
    for steps in episodes.values():
        game.reset(steps[0]["seed"])
        secret = game._secret
        for i, rec in enumerate(steps):
            prompt = rec["prompt_messages"][:2]  # clean system+user (drop retry turns)
            if not prompt:
                continue
            key = hashlib.sha1(
                json.dumps(prompt, sort_keys=True).encode()
            ).hexdigest()  # NOT state_hash: that is post-action and includes the sampled guess
            if key in seen:
                continue
            seen.add(key)
            prior = [
                [s["parsed_action"], *score_guess(s["parsed_action"], secret)]
                for s in steps[:i]
            ]
            ctx = {
                "secret": secret,
                "colors": game.colors,
                "dup_ok": game.allow_duplicates,
                "prior": prior,
            }
            rows.append((rec["step_idx"], {"prompt": prompt, "game_ctx": json.dumps(ctx)}))

    rows.sort(key=lambda t: t[0], reverse=True)  # later turns carry more feedback
    rows = rows[:MAX_PROMPTS]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for _, row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)
