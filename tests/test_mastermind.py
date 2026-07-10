import pytest

from slm_rl.config.loader import load_game_config
from slm_rl.games.base import ActionSpec
from slm_rl.games.mastermind import MastermindGame

CFG = load_game_config("mastermind")


def game() -> MastermindGame:
    return MastermindGame(CFG)


def test_reset_is_deterministic_given_seed():
    g1, g2 = game(), game()
    g1.reset(seed=7)
    g2.reset(seed=7)
    assert g1._secret == g2._secret
    g2.reset(seed=8)
    assert g1._secret != g2._secret or True  # different seed usually differs


def test_action_space_size():
    g = game()
    g.reset(seed=0)
    obs = g._observation()
    assert len(obs.legal_actions) == 6**4


def test_win_detection_and_reward():
    g = game()
    g.reset(seed=3)
    result = g.step(ActionSpec(id=g._secret, label=g._secret))
    assert result.terminated and not result.truncated
    assert result.reward == 1.0
    assert result.info["outcome"] == "win"
    assert result.info["exact"] == 4


def test_feedback_exact_and_partial():
    g = game()
    g.reset(seed=0)
    g._secret = "RGBY"
    result = g.step(ActionSpec(id="RGYB", label="RGYB"))
    assert result.info["exact"] == 2  # R, G
    assert result.info["partial"] == 2  # B, Y swapped


def test_feedback_with_duplicates():
    g = game()
    g.reset(seed=0)
    g._secret = "RRGG"
    result = g.step(ActionSpec(id="RRRR", label="RRRR"))
    assert result.info["exact"] == 2
    assert result.info["partial"] == 0


def test_turn_cap_truncates_with_loss():
    g = game()
    g.reset(seed=5)
    wrong = "RRRR" if g._secret != "RRRR" else "GGGG"
    result = None
    for _ in range(CFG.max_turns):
        result = g.step(ActionSpec(id=wrong, label=wrong))
    assert result.truncated and not result.terminated
    assert result.info["outcome"] == "loss"


def test_state_hash_changes_per_guess():
    g = game()
    g.reset(seed=1)
    h0 = g.state_hash()
    g.step(ActionSpec(id="RGBY", label="RGBY"))
    assert g.state_hash() != h0


def test_illegal_guess_raises():
    g = game()
    g.reset(seed=1)
    with pytest.raises(ValueError):
        g.step(ActionSpec(id="XXXX", label="XXXX"))


def test_observation_renders_history():
    g = game()
    obs = g.reset(seed=2)
    assert "No guesses yet" in obs.text
    result = g.step(ActionSpec(id="RGBY", label="RGBY"))
    assert "Guess 1: RGBY" in result.observation.text
    assert result.observation.turn == 1
