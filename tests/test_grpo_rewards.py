"""GRPO reward functions and entropy watchdog — pure, no model/GPU."""

import json
from types import SimpleNamespace

from slm_rl.training.grpo import EntropyFloorCallback, format_reward, return_reward


def ctx(**kwargs):
    base = {
        "legal_actions": ["UP", "DOWN", "FIRE"],
        "step_reward": 1.0,
        "discounted_return": 5.0,
        "target_action": "FIRE",
    }
    base.update(kwargs)
    return json.dumps(base)


def test_format_reward_tiers():
    completions = [
        "I have no idea what to do",
        "ACTION: LEFT",  # not in legal menu
        "ACTION: FIRE",
    ]
    assert format_reward(
        completions=completions, game_ctx=[ctx()] * 3
    ) == [-1.0, -0.5, 0.25]


def test_completions_may_be_message_lists():
    completions = [[{"role": "assistant", "content": "ACTION: FIRE"}]]
    assert format_reward(completions=completions, game_ctx=[ctx()]) == [0.25]


def test_return_reward_matches_target():
    completions = [
        "ACTION: FIRE",  # demonstrator -> full discounted return
        "ACTION: UP",    # legal other -> 0.25 * step_reward
        "gibberish",     # unparseable -> 0
    ]
    assert return_reward(
        completions=completions, game_ctx=[ctx()] * 3
    ) == [5.0, 0.25, 0.0]


def test_menu_index_resolution():
    c = ctx()
    assert format_reward(completions=["ACTION: 3"], game_ctx=[c]) == [0.25]
    assert return_reward(completions=["ACTION: 3"], game_ctx=[c]) == [5.0]
    assert format_reward(completions=["ACTION: 9"], game_ctx=[c]) == [-0.5]


def test_entropy_floor_stops_after_patience():
    cb = EntropyFloorCallback(floor=0.05, patience=3)
    control = SimpleNamespace(should_training_stop=False)
    for _ in range(2):
        cb.on_log(None, None, control, logs={"entropy": 0.01})
        assert not control.should_training_stop
    cb.on_log(None, None, control, logs={"entropy": 0.01})
    assert control.should_training_stop
    assert cb.collapsed
