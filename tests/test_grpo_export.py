"""export_grpo_dataset: discounted returns, prompt dedup, later-turn preference."""

import json

from slm_rl.config.schema import GameConfig
from slm_rl.datagen import grpo_export
from slm_rl.datagen.grpo_export import export_grpo_dataset

GAME_CFG = GameConfig(name="boxing")


def rec(ep, step, action, user_text, *, reward=1.0, generation=1, legal=None):
    return {
        "episode_id": ep,
        "step_idx": step,
        "seed": 0,
        "parsed_action": action,
        "reward": reward,
        "generation": generation,
        "legal_actions": legal or [{"id": "UP"}, {"id": "DOWN"}, {"id": "FIRE"}],
        "prompt_messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": user_text},
        ],
    }


def write_jsonl(tmp_path, records):
    p = tmp_path / "r.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records))
    return p


def test_discounted_returns_and_target(tmp_path):
    records = [
        rec("e1", 0, "UP", "turn 0", reward=1.0),
        rec("e1", 1, "FIRE", "turn 1", reward=2.0),
    ]
    out = tmp_path / "grpo.jsonl"
    n = export_grpo_dataset(write_jsonl(tmp_path, records), out, GAME_CFG)
    assert n == 2
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    by_step = {
        json.loads(r["game_ctx"])["target_action"]: json.loads(r["game_ctx"])
        for r in rows
    }
    # G1 = 2.0; G0 = 1 + 0.99*2 = 2.98
    assert by_step["FIRE"]["discounted_return"] == 2.0
    assert abs(by_step["UP"]["discounted_return"] - 2.98) < 1e-6
    assert by_step["UP"]["legal_actions"] == ["UP", "DOWN", "FIRE"]


def test_duplicate_prompts_deduped(tmp_path):
    records = [
        rec("e1", 0, "UP", "same text"),
        rec("e2", 0, "DOWN", "same text"),
    ]
    n = export_grpo_dataset(write_jsonl(tmp_path, records), tmp_path / "g.jsonl", GAME_CFG)
    assert n == 1


def test_cap_prefers_later_turns(tmp_path, monkeypatch):
    monkeypatch.setattr(grpo_export, "MAX_PROMPTS", 2)
    records = [rec("e1", i, "UP", f"turn {i}", reward=float(i)) for i in range(5)]
    out = tmp_path / "g.jsonl"
    assert export_grpo_dataset(write_jsonl(tmp_path, records), out, GAME_CFG) == 2
    steps = sorted(
        json.loads(json.loads(l)["game_ctx"]).get("step_reward", 0)
        for l in out.read_text().splitlines()
    )
    assert steps == [3.0, 4.0]


def test_max_prompts_kwarg_caps_rows(tmp_path):
    records = [rec("e1", i, "UP", f"turn {i}", reward=float(i)) for i in range(10)]
    out = tmp_path / "g.jsonl"
    assert export_grpo_dataset(
        write_jsonl(tmp_path, records), out, GAME_CFG, max_prompts=3
    ) == 3
    assert len(out.read_text().splitlines()) == 3


def test_cap_prefers_recent_generations(tmp_path, monkeypatch):
    monkeypatch.setattr(grpo_export, "MAX_PROMPTS", 2)
    records = [
        rec("old", 9, "UP", "old-a", generation=1),
        rec("old", 10, "UP", "old-b", generation=1),
        rec("new", 0, "FIRE", "new-a", generation=2),
        rec("new", 1, "FIRE", "new-b", generation=2),
    ]
    out = tmp_path / "g.jsonl"
    assert export_grpo_dataset(write_jsonl(tmp_path, records), out, GAME_CFG) == 2
    actions = {
        json.loads(json.loads(l)["game_ctx"])["target_action"]
        for l in out.read_text().splitlines()
    }
    assert actions == {"FIRE"}
