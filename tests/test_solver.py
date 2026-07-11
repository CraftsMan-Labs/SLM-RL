"""Mastermind exact-solver teacher: candidate filtering, win rate through the
real EpisodeRunner, LLM-identical prompts, monitor-clean episodes."""

import re
from pathlib import Path

from slm_rl.agents.llm_agent import extract_action_token
from slm_rl.config.loader import load_game_config
from slm_rl.datagen.schema import RolloutRecord
from slm_rl.datagen.writer import RolloutWriter
from slm_rl.games.mastermind.env import MastermindGame, score_guess
from slm_rl.rollout.runner import EpisodeRunner
from slm_rl.teachers import make_teacher
from slm_rl.teachers.mastermind_solver import consistent_candidates

CFG = load_game_config("mastermind")


def test_consistent_candidates_golden():
    # secret pinned to RGB?: two guesses agreeing on the first three pegs
    hist = [("RGBO", 3, 0), ("RGBP", 3, 0)]
    cands = consistent_candidates("RGBYOP", 4, True, hist)
    assert sorted(cands) == ["RGBB", "RGBG", "RGBR", "RGBY"]
    assert all(score_guess(g, c) == (e, p) for c in cands for g, e, p in hist)


def test_consistent_candidates_empty_history_is_full_space():
    assert len(consistent_candidates("RGBYOP", 4, True, [])) == 6**4
    assert len(consistent_candidates("RGBY", 2, False, [])) == 4 * 3


def test_played_guesses_are_always_excluded():
    # a non-winning guess is inconsistent with its own feedback:
    # score_guess(g, g) == (n, 0) which can't equal its real feedback
    game = MastermindGame(CFG)
    game.reset(seed=5)
    for guess in ("RRRR", "GGBB", "YOPY"):
        game.step(game._actions[[a.id for a in game._actions].index(guess)])
    hist = list(game._history)
    cands = consistent_candidates(game.colors, game.code_length, True, hist)
    played = {g for g, _, _ in hist}
    assert not played & set(cands)
    assert game._secret in cands  # the secret always survives


def test_solver_wins_and_is_deterministic():
    def play(n_episodes):
        agent, _ = make_teacher(CFG, seed=0)
        outcomes = []
        for i in range(n_episodes):
            runner = EpisodeRunner(MastermindGame(CFG), agent, CFG)
            summary = runner.run_episode(seed=1000 + i, episode_id=f"s-{i}")
            outcomes.append((summary["outcome"], summary["steps"]))
        return outcomes

    first = play(100)
    wins = sum(o == "win" for o, _ in first)
    assert wins / len(first) >= 0.95  # random-consistent wins standard mastermind
    assert first == play(100)  # seeded -> reproducible


def test_teacher_records_feed_sft_export(tmp_path: Path):
    from slm_rl.config.schema import TrainConfig
    from slm_rl.datagen.sft_export import export_sft_dataset

    agent, model_id = make_teacher(CFG, seed=0)
    assert model_id == "teacher:mastermind_solver"
    out = tmp_path / "teacher.jsonl"
    with RolloutWriter(out) as writer:
        for i in range(5):
            runner = EpisodeRunner(
                MastermindGame(CFG), agent, CFG, writer=writer, model_id=model_id
            )
            summary = runner.run_episode(seed=200 + i, episode_id=f"t-{i}")
            assert summary["monitor"]["interventions"] == 0  # monitor-clean

    records = [RolloutRecord.from_json(l) for l in out.read_text().splitlines()]
    # LLM-identical prompts: without these, export_sft_dataset yields 0 pairs
    assert all(len(r.prompt_messages) == 2 for r in records)
    assert all(r.prompt_messages[1]["content"].endswith("ACTION: <your move>") for r in records)
    assert all(r.model_id == "teacher:mastermind_solver" for r in records)

    pairs = export_sft_dataset(out, tmp_path / "sft.jsonl", TrainConfig())
    assert pairs > 0


def test_teacher_completion_verbalizes_rationale():
    # process supervision (plan 002): the completion states the deduction
    # (a number of remaining candidates) before the action line, and the
    # rationale never breaks the "last ACTION: line" parsing contract.
    agent, _ = make_teacher(CFG, seed=0)
    game = MastermindGame(CFG)
    obs = game.reset(seed=3)
    decision = agent.act(obs, [])
    completion = decision.raw_completion

    assert completion.count("ACTION:") == 1  # single occurrence, on the last line
    lines = completion.splitlines()
    assert lines[-1].startswith("ACTION:")
    assert re.search(r"\d", completion)  # candidate count is present

    guess = extract_action_token(completion)
    assert guess == decision.action.id

    # play a second turn so the history-driven rationale branch is exercised
    step = game.step(decision.action)
    decision2 = agent.act(step.observation, [])
    completion2 = decision2.raw_completion
    assert completion2.count("ACTION:") == 1
    assert completion2.splitlines()[-1] == f"ACTION: {decision2.action.id}"
    assert extract_action_token(completion2) == decision2.action.id
