"""Heuristic Space Invaders teacher (plan 009 + revision): beats random with
exploration on, episode diversity, per-construction determinism, rationale
contract, feeds SFT export, monitor-clean. Follows tests/test_solver.py
patterns. Skips entirely on machines without the [atari] extra."""

import re
from pathlib import Path

import pytest

pytest.importorskip("ale_py")
pytest.importorskip("gymnasium")

from slm_rl.agents.base import ActionDecision, Agent
from slm_rl.agents.bots import RandomAgent
from slm_rl.agents.llm_agent import extract_action_token
from slm_rl.config.loader import load_game_config
from slm_rl.datagen.schema import RolloutRecord
from slm_rl.datagen.writer import RolloutWriter
from slm_rl.games.base import Observation
from slm_rl.games.registry import get_game
from slm_rl.rollout.runner import EpisodeRunner
from slm_rl.teachers import make_teacher

CFG = load_game_config("space-invaders")


class _Recorder(Agent):
    """Pass-through wrapper capturing the action-id sequence per episode."""

    def __init__(self, inner: Agent):
        self.inner = inner
        self.actions: list[str] = []

    def act(self, obs: Observation) -> ActionDecision:
        decision = self.inner.act(obs)
        self.actions.append(decision.action.id)
        return decision


def _run_episodes(agent: Agent, seeds) -> tuple[list[float], list[tuple[str, ...]]]:
    """Consecutive episodes with ONE agent instance (the warm-start usage
    pattern: the RNG advances across episodes)."""
    rewards, sequences = [], []
    for i, seed in enumerate(seeds):
        recorder = _Recorder(agent)
        game = get_game("space-invaders")(CFG)
        runner = EpisodeRunner(game, recorder, CFG)
        summary = runner.run_episode(seed=seed, episode_id=f"e-{i}")
        rewards.append(summary["cum_reward"])
        sequences.append(tuple(recorder.actions))
    return rewards, sequences


def test_teacher_beats_random_with_exploration():
    # Exploration ON (EXPLORE_EPS is baked in): the teacher must clearly
    # beat the random baseline over the same seeds. Originally calibrated
    # at max_turns=80: teacher 2.8083 vs random 1.1167 (2.51x), asserted
    # >= 2x (~80% of measured). At natural episode length (max_turns=400,
    # stagnation window 240) BOTH agents run to game over, and random
    # benefits relatively more (it fires often enough to keep resetting
    # the stagnation counter, and long episodes give lucky hits time to
    # accumulate): recalibrated 2026-07-11 (agent seed 0, seeds 0-19) --
    # teacher 7.1833 vs random 3.8167 (1.88x). Same ~80%-of-measured
    # margin convention: assert >= 1.5x. The old 2x contrast was partly an
    # artifact of the window-40 monitor ejecting random players early.
    seeds = list(range(20))
    agent, _ = make_teacher(CFG, seed=0)
    teacher_rewards, _ = _run_episodes(agent, seeds)
    random_rewards, _ = _run_episodes(RandomAgent(seed=0), seeds)
    teacher_mean = sum(teacher_rewards) / len(teacher_rewards)
    random_mean = sum(random_rewards) / len(random_rewards)
    # random_mean can be non-positive (life loss penalties dominate); guard
    # the comparison so a >=1.5x claim is always meaningful.
    assert random_mean > 0, "random baseline must be positive for a 1.5x claim to mean anything"
    assert teacher_mean >= 1.5 * random_mean


def test_episode_diversity_across_consecutive_episodes():
    # The warm start plays ~1000 episodes with ONE agent; greedy play on
    # this env is seed-invariant, so diversity must come from the agent's
    # epsilon exploration. FAILS on the pre-epsilon code (measured: 1
    # distinct sequence over 10 episodes).
    agent, _ = make_teacher(CFG, seed=0)
    _, sequences = _run_episodes(agent, range(10))
    assert len(set(sequences)) >= 8


def test_determinism_per_construction():
    # The guarantee is per (agent seed, episode order) -- the RNG advances
    # across episodes -- NOT per episode seed alone: two agents built with
    # the same seed and run over the same episode seeds in the same order
    # produce identical action sequences.
    def run():
        agent, _ = make_teacher(CFG, seed=0)
        rewards, sequences = _run_episodes(agent, range(5))
        return rewards, sequences

    rewards_a, sequences_a = run()
    rewards_b, sequences_b = run()
    assert sequences_a == sequences_b
    assert rewards_a == rewards_b
    # and the sequences differ across episode positions (exploration active)
    assert len(set(sequences_a)) > 1


def test_rationale_contract():
    # Covers both branches (aim-and-fire and exploratory) over enough
    # decisions that epsilon draws occur with near-certainty is NOT assumed
    # here -- the contract must hold on every decision regardless of branch.
    game = get_game("space-invaders")(CFG)
    agent, _ = make_teacher(CFG, seed=0)
    obs = game.reset(seed=0)
    for _ in range(30):
        decision = agent.act(obs)
        completion = decision.raw_completion

        assert completion.count("ACTION:") == 1
        lines = completion.splitlines()
        assert lines[-1] == f"ACTION: {decision.action.id}"
        assert re.search(r"\d", completion)  # coordinates present
        assert extract_action_token(completion) == decision.action.id

        result = game.step(decision.action)
        obs = result.observation
        if result.terminated or result.truncated:
            break


def test_exploratory_rationale_is_honest():
    # A random move must never claim aim-based reasoning: drive the agent
    # until an exploratory decision occurs (eps=0.05 over 30+ decisions per
    # episode makes this near-certain within a few episodes) and check its
    # verbalization.
    agent, _ = make_teacher(CFG, seed=0)
    exploratory = []
    for seed in range(5):
        game = get_game("space-invaders")(CFG)
        obs = game.reset(seed=seed)
        while True:
            decision = agent.act(obs)
            if "to vary my approach" in decision.raw_completion:
                exploratory.append(decision)
            result = game.step(decision.action)
            obs = result.observation
            if result.terminated or result.truncated:
                break
        if exploratory:
            break
    assert exploratory, "no exploratory move in 5 episodes -- eps wiring broken?"
    for decision in exploratory:
        completion = decision.raw_completion
        assert "lined up" not in completion
        assert "moving that way" not in completion
        assert completion.count("ACTION:") == 1
        assert completion.splitlines()[-1] == f"ACTION: {decision.action.id}"


def test_teacher_records_feed_sft_export(tmp_path: Path):
    from slm_rl.config.schema import TrainConfig
    from slm_rl.datagen.sft_export import export_sft_dataset

    agent, model_id = make_teacher(CFG, seed=0)
    assert model_id == "teacher:space_invaders_heuristic"
    out = tmp_path / "teacher.jsonl"
    with RolloutWriter(out) as writer:
        for i, seed in enumerate(range(5)):
            game = get_game("space-invaders")(CFG)
            runner = EpisodeRunner(
                game, agent, CFG, writer=writer, model_id=model_id
            )
            runner.run_episode(seed=seed, episode_id=f"t-{i}")

    records = [RolloutRecord.from_json(l) for l in out.read_text().splitlines()]
    assert all(len(r.prompt_messages) == 2 for r in records)
    assert all(r.model_id == "teacher:space_invaders_heuristic" for r in records)

    pairs = export_sft_dataset(out, tmp_path / "sft.jsonl", TrainConfig())
    assert pairs > 0

    exported = (tmp_path / "sft.jsonl").read_text().splitlines()
    assert any("invader block" in line for line in exported)


def test_monitor_tolerance_no_repetition_flags():
    # The plan-009 concern: the heuristic legitimately repeats movement
    # actions (hold a direction, camp with FIRE), so NO repetition-type
    # signal may ever fire. At natural episode length (max_turns=400,
    # episodes end at game over rather than an 80-decision cap) competent
    # play produces long repeats AND long scoreless dry spells -- measured
    # 2026-07-11 (agent seeds 0+1 x env seeds 0-19, monitor bypassed): max
    # identical-action streak 44, max trailing 2-gram repeats 22, max
    # scoreless-decision streak 120 -- and the yaml thresholds (88/44/240)
    # are all calibrated to 2x those maxima, so a competent episode must
    # stay fully monitor-clean. This matters beyond hygiene: a single
    # reflect marks the whole episode dirty in sft_export.select_episodes
    # (dirty() excludes on ANY intervention), silently discarding the best
    # warm-start demonstrations. Measured 2026-07-11 at the calibrated
    # thresholds (deterministic: seeded agent + env, env seeds 0-9): 10 of
    # 10 episodes clean, zero signals, all ending at natural game over
    # (scores 120-515). Assert >= 9, the same 1-episode margin the
    # original used (measured 9, asserted >= 8); anything that fires
    # within that margin must be the stagnation guard -- the only
    # legitimate ejector (of never-scoring players) -- never a
    # repetition signal.
    agent, _ = make_teacher(CFG, seed=0)
    clean = 0
    for seed in range(10):
        game = get_game("space-invaders")(CFG)
        runner = EpisodeRunner(game, agent, CFG)
        summary = runner.run_episode(seed=seed, episode_id=f"m-{seed}")
        signals = summary["monitor"]["signals"]
        assert signals.get("action_repeat", 0) == 0
        assert signals.get("ngram_loop", 0) == 0
        assert signals.get("state_revisit", 0) == 0
        # anything that does fire must be the stagnation guard, nothing else
        assert set(signals) <= {"reward_stagnation"}
        if summary["monitor"]["interventions"] == 0:
            clean += 1
    assert clean >= 9
