"""Rollout parquet/JSONL -> GRPO prompt dataset.

Every decision step is usable: rewards are grounded in recorded step rewards
and discounted returns (no game-specific reconstruction). Each row carries
the prompt plus a `game_ctx` JSON column the reward functions score against.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from slm_rl.config.schema import GameConfig
from slm_rl.datagen.sft_export import group_episodes

MAX_PROMPTS = 512  # ponytail: fixed cap keeps GRPO generation minutes-long
GAMMA = 0.99


def _discounted_returns(rewards: list[float], gamma: float = GAMMA) -> list[float]:
    """Monte-Carlo returns G_t = r_t + gamma * G_{t+1}."""
    out = [0.0] * len(rewards)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = float(rewards[i]) + gamma * running
        out[i] = running
    return out


def _legal_ids(rec: dict) -> list[str]:
    menu = rec.get("legal_actions") or []
    ids: list[str] = []
    for item in menu:
        if isinstance(item, dict):
            aid = item.get("id")
            if aid:
                ids.append(str(aid))
        elif isinstance(item, str):
            ids.append(item)
    return ids


def export_grpo_dataset(
    dataset_path: Path,
    out_path: Path,
    game_cfg: GameConfig,
    *,
    max_prompts: int | None = None,
) -> int:
    """Write {"prompt", "game_ctx"} JSONL rows; return row count.

    `game_cfg` is accepted for API compatibility with TrainingStrategy; the
    generic exporter does not need game-specific reconstruction.
    `max_prompts` caps rows (None → module ``MAX_PROMPTS``; workshop uses 32).
    """
    del game_cfg  # unused — rewards use recorded fields only
    episodes = group_episodes(Path(dataset_path))
    # Resolve at call time so tests can monkeypatch MAX_PROMPTS.
    cap = max(1, int(MAX_PROMPTS if max_prompts is None else max_prompts))

    rows: list[tuple[tuple[int, int], dict]] = []
    seen: set[str] = set()
    for steps in episodes.values():
        rewards = [float(s.get("reward") or 0.0) for s in steps]
        returns = _discounted_returns(rewards)
        for i, rec in enumerate(steps):
            prompt = rec.get("prompt_messages") or []
            prompt = prompt[:2]  # clean system+user (drop retry turns)
            if not prompt:
                continue
            key = hashlib.sha1(
                json.dumps(prompt, sort_keys=True).encode()
            ).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            legal = _legal_ids(rec)
            ctx = {
                "legal_actions": legal,
                "step_reward": rewards[i],
                "discounted_return": returns[i],
                "target_action": rec.get("parsed_action"),
                "parse_status": rec.get("parse_status"),
            }
            rows.append((
                (int(rec.get("generation") or 0), int(rec.get("step_idx") or 0)),
                {"prompt": prompt, "game_ctx": json.dumps(ctx)},
            ))

    rows.sort(key=lambda t: t[0], reverse=True)
    rows = rows[:cap]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for _, row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)
