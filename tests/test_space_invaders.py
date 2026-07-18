"""Space Invaders via ALE (plan 008): registration, determinism, the 6-action
menu, reward/outcome plumbing, renderer budget, vector_obs, and a real
EpisodeRunner integration with the loosened doom-loop thresholds. Skips
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

CFG = load_game_config("space-invaders")


def test_registration_and_config_load():
    cls = get_game("space-invaders")
    assert cls.name == "space-invaders"
    assert CFG.extra["env_id"] == "ALE/SpaceInvaders-v5"


def test_determinism_same_seed_identical_hash_repeatably():
    # CODING_GUIDELINE Sec 1.4: same seed -> byte-identical decisions.
    # Note: verified empirically (plan 008) that with
    # repeat_action_probability=0.0, ALE's Space Invaders has NO
    # seed-conditioned randomness for a *fixed* action script (sticky
    # actions are the only per-seed RNG source in this game, and they are
    # deliberately disabled for determinism) -- a fixed 5-action script
    # produces byte-identical RAM regardless of seed. So this test asserts
    # same-seed repeatability only; it does not assert cross-seed
    # divergence under a fixed script, which would be a false expectation
    # for this particular ALE environment.
    script = [
        ActionSpec(id="FIRE", label="fire"),
        ActionSpec(id="RIGHT", label="move right"),
        ActionSpec(id="RIGHT", label="move right"),
        ActionSpec(id="FIRE", label="fire"),
        ActionSpec(id="LEFT", label="move left"),
    ]

    def run(seed):
        game = get_game("space-invaders")(CFG)
        game.reset(seed=seed)
        after_reset_hash = game.state_hash()
        for action in script:
            game.step(action)
        return after_reset_hash, game.state_hash()

    reset_hash_a, final_hash_a = run(0)
    reset_hash_b, final_hash_b = run(0)
    assert reset_hash_a == reset_hash_b
    assert final_hash_a == final_hash_b


def test_determinism_agent_driven_divergence_across_seeds():
    # The real seed-dependent variation in this pipeline comes from
    # RandomAgent's action *choices* (seeded), not from the ALE env itself
    # -- two RandomAgents seeded differently pick different actions and so
    # diverge, which is what a run's reproducibility actually depends on.
    # The first ~34 decisions are a fixed level-intro lockout during which
    # the cannon does not respond to input (verified empirically, plan
    # 008), so RAM stays identical across agent seeds until then; 20
    # decisions is past enough of that window to observe divergence.
    def run(seed):
        game = get_game("space-invaders")(CFG)
        agent = RandomAgent(seed=seed)
        obs = game.reset(seed=0)
        for _ in range(20):
            decision = agent.act(obs)
            result = game.step(decision.action)
            obs = result.observation
            if result.terminated or result.truncated:
                break
        return game.state_hash()

    assert run(0) != run(1)


def test_menu_has_six_actions_matching_env_meanings():
    game = get_game("space-invaders")(CFG)
    obs = game.reset(seed=0)
    assert len(obs.legal_actions) == 6
    ids = {a.id for a in obs.legal_actions}
    assert ids == {"NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"}
    # every id must resolve to a step without error
    for action in obs.legal_actions:
        game.step(action)


def test_reward_and_outcome_on_truncation_with_noop():
    # Standing still under NOOP never scores (no reward source without
    # firing), so reward is 0.0 unless a bomb kills the player (verified
    # empirically: seed=0 with pure NOOP loses all 3 lives to bombs by
    # decision ~7 at action_repeat=3 -- see plan 008 probe transcript for
    # the underlying frame-level timing). Assert the no-score invariant
    # only up to the first life loss, then assert the truncate/outcome
    # contract for the remainder of the episode.
    game = get_game("space-invaders")(CFG)
    obs = game.reset(seed=0)
    noop = next(a for a in obs.legal_actions if a.id == "NOOP")

    result = None
    lost_a_life = False
    for _ in range(CFG.max_turns):
        result = game.step(noop)
        if not lost_a_life and not (result.terminated or result.truncated):
            assert result.reward in (0.0, CFG.extra["life_loss_penalty"])
            if result.reward != 0.0:
                lost_a_life = True
        if result.terminated or result.truncated:
            break

    assert result.terminated or result.truncated
    assert result.info["outcome"].startswith("score:")


def test_renderer_budget_and_content():
    game = get_game("space-invaders")(CFG)
    obs = game.reset(seed=0)
    lines = obs.text.splitlines()
    assert len(lines) <= 12
    assert len(obs.text) < 600
    assert "Lives" in obs.text
    assert "cannon" in obs.text


def test_observation_metadata_carries_decoded_state():
    # plan 009 Step 1: the renderer's decode(raw_obs) is duck-typed into
    # metadata["state"] so non-LLM consumers (the heuristic teacher) can
    # read named RAM variables without re-decoding.
    game = get_game("space-invaders")(CFG)
    obs = game.reset(seed=0)
    state = obs.metadata["state"]
    assert isinstance(state["player_x"], int)
    assert isinstance(state["enemies_x"], int)
    assert isinstance(state["invaders_left"], int)
    assert isinstance(state["missile_in_flight"], bool)


def test_vector_obs_length_and_range():
    game = get_game("space-invaders")(CFG)
    game.reset(seed=0)
    vec = game.vector_obs()
    assert len(vec) == 128
    assert all(0.0 <= v <= 1.0 for v in vec)


def _cfg_with_noop(noop_start_max: int):
    extra = dict(CFG.extra)
    extra["noop_start_max"] = noop_start_max
    return CFG.model_copy(update={"extra": extra})


def test_noop_start_makes_different_seeds_diverge_at_reset():
    # Motivation (measured 2026-07-11, run spaceinv-350m-v2): with
    # repeat_action_probability=0.0, ALE Space Invaders is seed-invariant at
    # reset -- confirmed on the pre-change adapter (stash-style probe):
    # reset(seed=1) and reset(seed=2) produced the IDENTICAL state_hash
    # (f0b0916d521bdb1c for both). Random no-op starts (Mnih et al. 2015 DQN
    # eval protocol), seeded per episode, fix this: different seeds must now
    # diverge at reset.
    cfg = _cfg_with_noop(30)

    def reset_hash(seed):
        game = get_game("space-invaders")(cfg)
        game.reset(seed=seed)
        return game.state_hash()

    assert reset_hash(1) != reset_hash(2)


def test_noop_start_same_seed_reproducible():
    # CODING_GUIDELINE Sec 1.4: same seed -> byte-identical decisions, even
    # with the noop-start RNG in play (k = f(seed) is deterministic).
    cfg = _cfg_with_noop(30)

    def reset_hash(seed):
        game = get_game("space-invaders")(cfg)
        game.reset(seed=seed)
        return game.state_hash()

    assert reset_hash(7) == reset_hash(7)


def test_noop_start_disabled_matches_no_key_present():
    # New knobs default to "current behavior unchanged" (CODING_GUIDELINE
    # Sec 2): noop_start_max=0 (explicit) must produce the exact same reset
    # state as a config that omits the key entirely.
    # yaml now has noop_start_max: 30, so build a variant that omits the key
    extra_no_key = {k: v for k, v in CFG.extra.items() if k != "noop_start_max"}
    cfg_absent = CFG.model_copy(update={"extra": extra_no_key})
    cfg_zero = _cfg_with_noop(0)

    def reset_hash(cfg, seed):
        game = get_game("space-invaders")(cfg)
        game.reset(seed=seed)
        return game.state_hash()

    for seed in (0, 1, 2):
        assert reset_hash(cfg_absent, seed) == reset_hash(cfg_zero, seed)


def test_episode_runner_integration_random_agent_two_seeds(tmp_path):
    # Records written, no crash; the game itself always reports a "score:"
    # outcome when it terminates/truncates on its own timeline (max_turns).
    # If the doom-loop monitor truncates the episode first, `outcome` falls
    # back to the runner's literal "truncated" -- that is fine as long as it
    # doesn't happen prematurely, which is the actual regression this test
    # guards (loosened thresholds, see configs/games/space-invaders.yaml):
    # a random (non-NOOP-only) script must not be monitor-truncated before
    # decision 20.
    out = tmp_path / "space-invaders.jsonl"
    with RolloutWriter(out) as writer:
        for i, seed in enumerate((10_000, 10_001)):
            game = get_game("space-invaders")(CFG)
            agent = RandomAgent(seed=seed)
            runner = EpisodeRunner(game, agent, CFG, writer=writer)
            summary = runner.run_episode(seed=seed, episode_id=f"si-{i}")
            assert isinstance(summary["outcome"], str) and summary["outcome"]
            assert summary["steps"] >= 20

    lines = out.read_text().splitlines()
    assert len(lines) > 0
