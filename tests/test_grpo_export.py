"""export_grpo_dataset: secret reconstruction, prior feedback, prompt dedup,
later-turn preference under the cap."""

import json

import pytest

from slm_rl.config.schema import GameConfig
from slm_rl.datagen import grpo_export
from slm_rl.datagen.grpo_export import export_grpo_dataset
from slm_rl.games.mastermind.env import MastermindGame, score_guess

GAME_CFG = GameConfig(name="mastermind")


def rec(ep, step, seed, action, user_text):
    return {
        "episode_id": ep, "step_idx": step, "seed": seed, "parsed_action": action,
        "prompt_messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": user_text},
        ],
    }


def write_jsonl(tmp_path, records):
    p = tmp_path / "r.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records))
    return p


def test_secret_and_prior_reconstructed(tmp_path):
    seed = 42
    secret = MastermindGame(GAME_CFG)
    secret.reset(seed)
    secret = secret._secret

    records = [
        rec("e1", 0, seed, "RRRR", "turn 0"),
        rec("e1", 1, seed, "GGGG", "turn 1"),
    ]
    out = tmp_path / "grpo.jsonl"
    n = export_grpo_dataset(write_jsonl(tmp_path, records), out, GAME_CFG)
    assert n == 2

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    rows.sort(key=lambda r: len(json.loads(r["game_ctx"])["prior"]))
    ctx0 = json.loads(rows[0]["game_ctx"])
    ctx1 = json.loads(rows[1]["game_ctx"])
    assert ctx0["secret"] == secret and ctx0["prior"] == []
    exact, partial = score_guess("RRRR", secret)
    assert ctx1["prior"] == [["RRRR", exact, partial]]
    assert len(rows[0]["prompt"]) == 2  # clean system+user


def test_duplicate_prompts_deduped(tmp_path):
    records = [
        rec("e1", 0, 1, "RRRR", "same text"),
        rec("e2", 0, 2, "GGGG", "same text"),  # identical prompt, different episode
    ]
    n = export_grpo_dataset(write_jsonl(tmp_path, records), tmp_path / "g.jsonl", GAME_CFG)
    assert n == 1


def test_cap_prefers_later_turns(tmp_path, monkeypatch):
    monkeypatch.setattr(grpo_export, "MAX_PROMPTS", 2)
    records = [rec("e1", i, 7, "RRRR", f"turn {i}") for i in range(5)]
    out = tmp_path / "g.jsonl"
    assert export_grpo_dataset(write_jsonl(tmp_path, records), out, GAME_CFG) == 2
    priors = [len(json.loads(json.loads(l)["game_ctx"])["prior"]) for l in out.read_text().splitlines()]
    assert sorted(priors) == [3, 4]  # the two latest turns kept


def test_non_mastermind_rejected(tmp_path):
    with pytest.raises(NotImplementedError):
        export_grpo_dataset(tmp_path, tmp_path / "g.jsonl", GameConfig(name="connect4"))


def test_menu_stamped_only_for_pruned_prompts(tmp_path):
    pruned = rec("e1", 0, 1, "RRRR", "pruned prompt")
    pruned["legal_actions"] = ["GGGG", "RRRR", "YYBB"]  # <= MENU_LIMIT
    full = rec("e2", 0, 2, "GGGG", "format prompt")
    full["legal_actions"] = [f"A{i:04d}" for i in range(40)]  # > MENU_LIMIT
    out = tmp_path / "g.jsonl"
    assert export_grpo_dataset(write_jsonl(tmp_path, [pruned, full]), out, GAME_CFG) == 2

    rows = [json.loads(l) for l in out.read_text().splitlines()]
    by_menu = [json.loads(r["game_ctx"]).get("menu") for r in rows]
    assert sorted(x is not None for x in by_menu) == [False, True]
    menu = next(x for x in by_menu if x is not None)
    assert menu == ["GGGG", "RRRR", "YYBB"]  # rollout order preserved (1-indexed by rewards)
