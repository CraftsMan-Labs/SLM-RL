"""GRPO reward functions, entropy watchdog, and strict parsing — all pure,
no model/GPU."""

import json
from types import SimpleNamespace

from slm_rl.agents.llm_agent import extract_action_token, parse_action
from slm_rl.games.base import ActionSpec
from slm_rl.training.grpo import EntropyFloorCallback, consistency_reward, format_reward

CTX = json.dumps({"secret": "RGBY", "colors": "RGBYOP", "dup_ok": True, "prior": []})


def ctx_with_prior(*prior):
    return json.dumps(
        {"secret": "RGBY", "colors": "RGBYOP", "dup_ok": True, "prior": list(prior)}
    )


def test_format_reward_tiers():
    completions = [
        "I have no idea what to do",       # unparseable -> -1
        "ACTION: RGBX",                    # X not a color -> -0.5
        "ACTION: RGB",                     # wrong length -> -0.5
        "ACTION: RGBY",                    # legal -> +0.25
    ]
    assert format_reward(completions=completions, game_ctx=[CTX] * 4) == [-1.0, -0.5, -0.5, 0.25]


def test_completions_may_be_message_lists():
    completions = [[{"role": "assistant", "content": "ACTION: RGBY"}]]
    assert format_reward(completions=completions, game_ctx=[CTX]) == [0.25]


def test_consistency_fraction_and_secret_bonus():
    # secret RGBY; prior guess RRRR scored (1 exact, 0 partial)
    ctx = ctx_with_prior(["RRRR", 1, 0])
    completions = [
        "ACTION: GGGG",   # feedback vs RRRR would be (0,0) != (1,0) -> inconsistent
        "ACTION: GBYR",   # feedback vs RRRR is (1,0) -> consistent
        "ACTION: RGBY",   # consistent AND the secret -> 1 + 1
        "gibberish",      # unparseable -> 0 (format_reward penalizes)
    ]
    assert consistency_reward(completions=completions, game_ctx=[ctx] * 4) == [0.0, 1.0, 2.0, 0.0]


def test_consistency_empty_history_is_zero():
    assert consistency_reward(completions=["ACTION: OOOO"], game_ctx=[CTX]) == [0.0]


def test_entropy_callback_trips_after_patience():
    cb = EntropyFloorCallback(floor=0.3, patience=3)
    control = SimpleNamespace(should_training_stop=False)
    for entropy in [0.5, 0.1, 0.1]:
        cb.on_log(None, None, control, logs={"entropy": entropy})
    assert not cb.collapsed
    cb.on_log(None, None, control, logs={"entropy": 0.1})
    assert cb.collapsed and control.should_training_stop


def test_entropy_callback_streak_resets():
    cb = EntropyFloorCallback(floor=0.3, patience=2)
    control = SimpleNamespace(should_training_stop=False)
    for entropy in [0.1, 0.5, 0.1, 0.5, 0.1]:
        cb.on_log(None, None, control, logs={"entropy": entropy})
    assert not cb.collapsed


LEGAL = [ActionSpec(id="RGBY", label="RGBY"), ActionSpec(id="GGGG", label="GGGG")]


def test_strict_parse_drops_last_resort():
    text = "I would probably try RGBY here"  # no ACTION: line
    assert parse_action(text, LEGAL) is not None       # lenient rollout ladder
    assert parse_action(text, LEGAL, strict=True) is None


def test_empty_action_line_does_not_crash():
    # regex backtracking can capture whitespace-only -> used to IndexError
    assert extract_action_token("ACTION:  ") is None
    assert extract_action_token("ACTION: ``") is None  # strips to empty
    assert parse_action("ACTION:  ", LEGAL, strict=True) is None
