"""Reward hook seam in GymnasiumGameAdapter (plan 013): absent -> byte-
identical to before the knob existed; present -> wraps the built-in
formula's result. Skips entirely on machines without the [atari] extra."""

from __future__ import annotations

import pytest

pytest.importorskip("ale_py")
pytest.importorskip("gymnasium")

from slm_rl.config.loader import load_game_config
from slm_rl.games.base import ActionSpec
from slm_rl.games.registry import get_game

CFG = load_game_config("space-invaders")

# A fixed 40-decision action script -- deterministic, no RandomAgent
# involved (CODING_GUIDELINE Sec 3: never touch a real model; this test
# only drives the ALE env directly).
SCRIPT = [
    ActionSpec(id="FIRE", label="fire"),
    ActionSpec(id="RIGHT", label="move right"),
    ActionSpec(id="LEFT", label="move left"),
    ActionSpec(id="NOOP", label="noop"),
] * 10


def _run(cfg, seed=0):
    game = get_game("space-invaders")(cfg)
    game.reset(seed=seed)
    rewards = []
    for action in SCRIPT:
        result = game.step(action)
        rewards.append(result.reward)
        if result.terminated or result.truncated:
            break
    return rewards


def _cfg_with_extra(extra_overrides: dict):
    extra = dict(CFG.extra)
    extra.update(extra_overrides)
    return CFG.model_copy(update={"extra": extra})


def test_hook_absent_is_byte_identical_across_two_runs():
    rewards_a = _run(CFG)
    rewards_b = _run(CFG)
    assert rewards_a == rewards_b
    assert len(rewards_a) == 40


def test_hook_absent_vs_extra_without_key_is_byte_identical():
    # extra dict present but without "reward_hook" -- must behave exactly
    # like CFG (no key at all).
    cfg_no_key = _cfg_with_extra({})
    assert _run(CFG) == _run(cfg_no_key)


def test_hook_returning_default_reward_is_byte_identical(tmp_path):
    # Proves ctx carries the TRUE default: a hook that just echoes
    # ctx["default_reward"] must reproduce the no-hook reward sequence
    # exactly.
    hook_path = tmp_path / "identity_hook.py"
    hook_path.write_text(
        "def shape_reward(ctx):\n    return ctx['default_reward']\n",
        encoding="utf-8",
    )
    cfg_hook = _cfg_with_extra({"reward_hook": str(hook_path)})
    assert _run(CFG) == _run(cfg_hook)


def test_hook_doubling_default_reward_doubles_every_reward(tmp_path):
    hook_path = tmp_path / "double_hook.py"
    hook_path.write_text(
        "def shape_reward(ctx):\n    return 2 * ctx['default_reward']\n",
        encoding="utf-8",
    )
    cfg_hook = _cfg_with_extra({"reward_hook": str(hook_path)})

    baseline = _run(CFG)
    doubled = _run(cfg_hook)
    assert len(baseline) == len(doubled)
    for b, d in zip(baseline, doubled):
        assert d == pytest.approx(2 * b)


def test_missing_hook_path_raises_at_construction(tmp_path):
    missing = tmp_path / "does_not_exist.py"
    cfg_hook = _cfg_with_extra({"reward_hook": str(missing)})
    with pytest.raises(ValueError, match="reward_hook not found"):
        get_game("space-invaders")(cfg_hook)


def test_module_without_shape_reward_raises_on_first_step(tmp_path):
    hook_path = tmp_path / "no_shape_reward.py"
    hook_path.write_text("X = 1\n", encoding="utf-8")
    cfg_hook = _cfg_with_extra({"reward_hook": str(hook_path)})

    game = get_game("space-invaders")(cfg_hook)
    game.reset(seed=0)  # construction/reset succeed; the module loads lazily
    with pytest.raises(ValueError, match="shape_reward"):
        game.step(SCRIPT[0])
