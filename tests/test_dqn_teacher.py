"""CleanRL-pattern DQN teacher (plan 012): training mechanics, checkpoint
round-trip, rationale contract, determinism, diversity, factory wiring, and
the vector_obs metadata addition. Mirrors tests/test_si_teacher.py's
patterns. Decision counts are kept tiny (mechanics-only, no quality claim)
so the whole file stays well under the ~90s time-box."""

from __future__ import annotations

import math
import re

import pytest

pytest.importorskip("torch")
pytest.importorskip("ale_py")

from slm_rl.config.loader import load_game_config
from slm_rl.agents.llm_agent import extract_action_token
from slm_rl.games.registry import get_game
from slm_rl.teachers import make_teacher
from slm_rl.teachers.dqn import DQNTeacherAgent, train_dqn

CFG = load_game_config("space-invaders")

# Training mechanics only -- small enough to keep this file's total runtime
# well under 90s (measured ~5s for 1500 decisions on CPU).
TRAIN_DECISIONS = 1500


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("dqn") / "teacher.pt"
    summary = train_dqn(CFG, decisions=TRAIN_DECISIONS, out_path=out, device="cpu", seed=0, log_every=10_000)
    assert out.exists()
    assert summary["episodes"] >= 2
    assert summary["loss"] is not None and math.isfinite(summary["loss"])
    assert summary["buffer_size"] > 0
    return str(out)


def test_training_mechanics(checkpoint):
    # fixture assertions cover existence/finiteness/buffer/episode count;
    # this test exists so the requirement is visible as its own case.
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["decisions"] == TRAIN_DECISIONS
    assert payload["obs_dim"] == 128  # ALE RAM bytes
    assert payload["action_ids"]
    assert math.isfinite(payload["mean_ep_reward_last20"])


def test_checkpoint_roundtrip_act(checkpoint):
    agent = DQNTeacherAgent(checkpoint, "system prompt", seed=0)
    game = get_game("space-invaders")(CFG)
    obs = game.reset(seed=0)
    decision = agent.act(obs, [])
    legal_ids = {a.id for a in obs.legal_actions}
    assert decision.action.id in legal_ids


def test_rationale_contract(checkpoint):
    # mirrors test_si_teacher.py's contract: single ACTION: occurrence, last
    # line, digits present, extract_action_token recovers the id -- and over
    # enough decisions both greedy and exploratory rationales appear, with
    # exploratory ones never claiming Q-value reasoning. EXPLORE_EPS=0.05
    # means ~1.5 exploratory draws per 30 decisions in expectation, so drive
    # several episodes (like test_si_teacher.py's honesty test) rather than
    # one, to make a miss near-impossible without inflating the decision
    # count that would blow the file's time-box.
    agent = DQNTeacherAgent(checkpoint, "system prompt", seed=0)
    saw_greedy = saw_exploratory = False
    for ep_seed in range(8):
        game = get_game("space-invaders")(CFG)
        obs = game.reset(seed=ep_seed)
        history = []
        for _ in range(30):
            decision = agent.act(obs, history)
            completion = decision.raw_completion

            assert completion.count("ACTION:") == 1
            lines = completion.splitlines()
            assert lines[-1] == f"ACTION: {decision.action.id}"
            assert extract_action_token(completion) == decision.action.id

            if "Q-values rank" in completion:
                saw_greedy = True
                # digits present in the Q-value rationale (same contract as
                # 009's tests) -- the exploratory branch's literal template
                # ("Trying {id} to vary my approach.") carries no coordinate,
                # unlike 009's heuristic, so the digit check is per-branch.
                assert re.search(r"\d", completion)
            if "to vary my approach" in completion:
                saw_exploratory = True
                assert "Q-values rank" not in completion

            history.append(decision)
            result = game.step(decision.action)
            obs = result.observation
            if result.terminated or result.truncated:
                break
        if saw_greedy and saw_exploratory:
            break

    assert saw_greedy, "no greedy decision across episodes -- EXPLORE_EPS miscalibrated?"
    assert saw_exploratory, "no exploratory decision across episodes -- EXPLORE_EPS miscalibrated?"


def test_determinism_per_construction(checkpoint):
    def run():
        agent = DQNTeacherAgent(checkpoint, "system prompt", seed=0)
        actions = []
        for ep_seed in range(3):
            game = get_game("space-invaders")(CFG)
            obs = game.reset(seed=ep_seed)
            history = []
            for _ in range(15):
                decision = agent.act(obs, history)
                actions.append(decision.action.id)
                history.append(decision)
                result = game.step(decision.action)
                obs = result.observation
                if result.terminated or result.truncated:
                    break
        return actions

    assert run() == run()


def test_diversity_across_consecutive_episodes(checkpoint):
    agent = DQNTeacherAgent(checkpoint, "system prompt", seed=0)
    sequences = []
    for ep_seed in range(6):
        game = get_game("space-invaders")(CFG)
        obs = game.reset(seed=ep_seed)
        history = []
        actions = []
        for _ in range(15):
            decision = agent.act(obs, history)
            actions.append(decision.action.id)
            history.append(decision)
            result = game.step(decision.action)
            obs = result.observation
            if result.terminated or result.truncated:
                break
        sequences.append(tuple(actions))
    assert len(set(sequences)) >= 5


def test_factory_dqn_and_heuristic_and_error(checkpoint):
    agent, model_id = make_teacher(CFG, seed=0, dqn_checkpoint=checkpoint)
    assert isinstance(agent, DQNTeacherAgent)
    assert model_id == "teacher:space_invaders_dqn"

    agent2, model_id2 = make_teacher(CFG, seed=0)
    assert model_id2 == "teacher:space_invaders_heuristic"

    with pytest.raises(ValueError):
        make_teacher(CFG, seed=0, dqn_checkpoint="/nonexistent/path/checkpoint.pt")


def test_vector_obs_metadata_exposed():
    game = get_game("space-invaders")(CFG)
    obs = game.reset(seed=0)
    vec = obs.metadata["vector_obs"]
    assert len(vec) == 128
    assert all(0.0 <= v <= 1.0 for v in vec)
