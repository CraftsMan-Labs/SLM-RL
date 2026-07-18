"""Freeway via ALE (plan 016): registration, determinism, the 3-action menu,
reward/outcome plumbing, renderer budget, vector_obs, and a real
EpisodeRunner integration with the measured doom-loop thresholds. Skips
entirely on machines without the [atari] extra."""

import pytest

pytest.importorskip("ale_py")
pytest.importorskip("gymnasium")

from slm_rl.agents.bots import RandomAgent
from slm_rl.config.loader import load_game_config
from slm_rl.datagen.writer import RolloutWriter
from slm_rl.games.base import ActionSpec
from slm_rl.games.registry import get_game
from slm_rl.rollout.runner import EpisodeRunner

CFG = load_game_config("freeway")


def test_registration_and_config_load():
    cls = get_game("freeway")
    assert cls.name == "freeway"
    assert CFG.extra["env_id"] == "ALE/Freeway-v5"


def test_atari_freeway_no_longer_resolves():
    # plan 016 decision 1: the dead atari_freeway stub is removed, replaced
    # by the real "freeway" registry name.
    from slm_rl.games.registry import available_games

    assert "atari_freeway" not in available_games()
    assert "freeway" in available_games()


def test_determinism_same_seed_identical_hash_repeatably():
    # CODING_GUIDELINE Sec 1.4: same seed -> byte-identical decisions.
    script = [
        ActionSpec(id="UP", label="move up (toward the goal)"),
        ActionSpec(id="UP", label="move up (toward the goal)"),
        ActionSpec(id="DOWN", label="move down (retreat)"),
        ActionSpec(id="NOOP", label="stand still"),
        ActionSpec(id="UP", label="move up (toward the goal)"),
    ]

    def run(seed):
        game = get_game("freeway")(CFG)
        game.reset(seed=seed)
        after_reset_hash = game.state_hash()
        for action in script:
            game.step(action)
        return after_reset_hash, game.state_hash()

    reset_hash_a, final_hash_a = run(0)
    reset_hash_b, final_hash_b = run(0)
    assert reset_hash_a == reset_hash_b
    assert final_hash_a == final_hash_b


def test_noop_start_makes_different_seeds_diverge_at_reset():
    def reset_hash(seed):
        game = get_game("freeway")(CFG)
        game.reset(seed=seed)
        return game.state_hash()

    assert reset_hash(1) != reset_hash(2)


def test_noop_start_same_seed_reproducible():
    def reset_hash(seed):
        game = get_game("freeway")(CFG)
        game.reset(seed=seed)
        return game.state_hash()

    assert reset_hash(7) == reset_hash(7)


def _cfg_with_noop(noop_start_max: int):
    extra = dict(CFG.extra)
    extra["noop_start_max"] = noop_start_max
    return CFG.model_copy(update={"extra": extra})


def test_noop_start_disabled_matches_no_key_present():
    extra_no_key = {k: v for k, v in CFG.extra.items() if k != "noop_start_max"}
    cfg_absent = CFG.model_copy(update={"extra": extra_no_key})
    cfg_zero = _cfg_with_noop(0)

    def reset_hash(cfg, seed):
        game = get_game("freeway")(cfg)
        game.reset(seed=seed)
        return game.state_hash()

    for seed in (0, 1, 2):
        assert reset_hash(cfg_absent, seed) == reset_hash(cfg_zero, seed)


def test_menu_has_three_actions_matching_env_meanings():
    game = get_game("freeway")(CFG)
    obs = game.reset(seed=0)
    assert len(obs.legal_actions) == 3
    ids = {a.id for a in obs.legal_actions}
    assert ids == {"NOOP", "UP", "DOWN"}
    for action in obs.legal_actions:
        game.step(action)


def test_renderer_budget_and_content():
    game = get_game("freeway")(CFG)
    obs = game.reset(seed=0)
    lines = obs.text.splitlines()
    assert len(lines) <= 12
    assert len(obs.text) < 600
    assert "crossing" in obs.text.lower()
    assert "chicken" in obs.text.lower()


def test_observation_metadata_carries_decoded_state():
    game = get_game("freeway")(CFG)
    obs = game.reset(seed=0)
    state = obs.metadata["state"]
    assert isinstance(state["player_y"], int)
    assert isinstance(state["car_x"], list)
    assert len(state["car_x"]) == 10
    assert all(isinstance(x, int) for x in state["car_x"])


def test_vector_obs_length_and_range():
    game = get_game("freeway")(CFG)
    game.reset(seed=0)
    vec = game.vector_obs()
    assert len(vec) == 128
    assert all(0.0 <= v <= 1.0 for v in vec)


def test_score_is_never_negative():
    # Freeway score (crossing count) is always non-negative.
    game = get_game("freeway")(CFG)
    obs = game.reset(seed=0)
    up = next(a for a in obs.legal_actions if a.id == "UP")
    result = None
    for _ in range(200):
        result = game.step(up)
        assert result.reward >= 0.0
        if result.terminated or result.truncated:
            break


def test_episode_runner_integration_random_agent_two_seeds(tmp_path):
    out = tmp_path / "freeway.jsonl"
    with RolloutWriter(out) as writer:
        for i, seed in enumerate((10_000, 10_001)):
            game = get_game("freeway")(CFG)
            agent = RandomAgent(seed=seed)
            runner = EpisodeRunner(game, agent, CFG, writer=writer)
            summary = runner.run_episode(seed=seed, episode_id=f"freeway-{i}")
            assert isinstance(summary["outcome"], str) and summary["outcome"]
            assert summary["steps"] >= 20

    lines = out.read_text().splitlines()
    assert len(lines) > 0
