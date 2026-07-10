"""GRPO reward functions, entropy watchdog, and strict parsing — all pure,
no model/GPU."""

import json
from types import SimpleNamespace

from slm_rl.agents.llm_agent import extract_action_token, parse_action
from slm_rl.games.base import ActionSpec
from slm_rl.training.grpo import EntropyFloorCallback, deduction_reward, format_reward

CTX = json.dumps({"secret": "RGBY", "colors": "RGBYOP", "dup_ok": True, "prior": []})


def small_ctx(*prior, menu=None):
    """2-peg, 2-color game: candidate space {RR, RG, GR, GG}, secret RG —
    every elimination value is hand-computable."""
    ctx = {"secret": "RG", "colors": "RG", "dup_ok": True, "prior": list(prior)}
    if menu is not None:
        ctx["menu"] = menu
    return json.dumps(ctx)


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


def test_deduction_elimination_and_secret_bonus():
    # secret RG. Feedback (0,2) from GR pins the set to {RG} -> r=1.0;
    # GG's (1,0) leaves {RG, GR} -> r=0.5; the secret adds +1.0 on top.
    completions = [
        "ACTION: GR",   # eliminates 4 -> 1: (log4 - log1)/log4 = 1.0
        "ACTION: GG",   # leaves {RG, GR}:   (log4 - log2)/log4 = 0.5
        "ACTION: RG",   # secret hit: elimination 1.0 + bonus 1.0
        "gibberish",    # unparseable -> 0 (format_reward penalizes)
    ]
    rewards = deduction_reward(completions=completions, game_ctx=[small_ctx()] * 4)
    assert rewards == [1.0, 0.5, 2.0, 0.0]
    # informative > uninformative, and empty-history groups differentiate
    # (the old consistency fraction was constant 0.0 on turn 0)
    assert rewards[0] > rewards[1] > 0.0


def test_deduction_repeat_penalty():
    # the old consistency reward scored a repeated wrong guess (k-1)/k —
    # near-max — which is why GRPO never killed the repeat doom loop
    ctx = small_ctx(["GG", 1, 0])
    assert deduction_reward(completions=["ACTION: GG"], game_ctx=[ctx]) == [-1.0]


def test_deduction_single_candidate_guard():
    # prior pins the secret: |before| == 1, so elimination is undefined ->
    # 0.0 unless the guess IS the secret (then just the +1.0 hit bonus)
    ctx = small_ctx(["GR", 0, 2])  # only RG remains consistent
    assert deduction_reward(completions=["ACTION: RG"], game_ctx=[ctx]) == [1.0]
    assert deduction_reward(completions=["ACTION: RR"], game_ctx=[ctx]) == [0.0]


def test_menu_index_resolution():
    ctx = small_ctx(menu=["GG", "RG", "GR"])
    completions = ["ACTION: 1", "ACTION: 2", "ACTION: 3"]
    # an all-consistent (pruned) menu must still differentiate the group —
    # under the old consistency fraction all three scored 1.0 (dead gradient)
    assert deduction_reward(completions=completions, game_ctx=[ctx] * 3) == [0.5, 2.0, 1.0]
    assert format_reward(completions=["ACTION: 2"], game_ctx=[ctx]) == [0.25]


def test_menu_illegal_tokens():
    ctx = small_ctx(menu=["GG", "RG"])
    assert format_reward(completions=["ACTION: 9"], game_ctx=[ctx]) == [-0.5]   # out of range
    assert format_reward(completions=["ACTION: RR"], game_ctx=[ctx]) == [-0.5]  # off-menu code
    # no menu: a bare index has nothing to resolve against (format mode)
    assert format_reward(completions=["ACTION: 2"], game_ctx=[small_ctx()]) == [-0.5]


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
