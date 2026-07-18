"""Heuristic Freeway teacher (plan 016): pure hold-UP (car-avoidance did not
verify, see ram_maps/freeway.py + teachers/freeway.py docstrings for why),
feeds SFT export, monitor-clean at the shipped (effectively-off) repetition
thresholds. Skips entirely on machines without the [atari] extra."""

from pathlib import Path

import pytest

pytest.importorskip("ale_py")
pytest.importorskip("gymnasium")

from slm_rl.agents.llm_agent import extract_action_token
from slm_rl.config.loader import load_game_config
from slm_rl.datagen.schema import RolloutRecord
from slm_rl.datagen.writer import RolloutWriter
from slm_rl.games.registry import get_game
from slm_rl.rollout.runner import EpisodeRunner
from slm_rl.teachers import make_teacher

CFG = load_game_config("freeway")


def test_teacher_always_moves_up():
    game = get_game("freeway")(CFG)
    agent, _ = make_teacher(CFG, seed=0)
    obs = game.reset(seed=0)
    for _ in range(20):
        decision = agent.act(obs)
        assert decision.action.id == "UP"
        result = game.step(decision.action)
        obs = result.observation
        if result.terminated or result.truncated:
            break


def test_determinism_per_construction():
    def run():
        agent, _ = make_teacher(CFG, seed=0)
        game = get_game("freeway")(CFG)
        obs = game.reset(seed=0)
        actions = []
        for _ in range(30):
            decision = agent.act(obs)
            actions.append(decision.action.id)
            result = game.step(decision.action)
            obs = result.observation
            if result.terminated or result.truncated:
                break
        return actions

    assert run() == run()


def test_rationale_contract():
    game = get_game("freeway")(CFG)
    agent, _ = make_teacher(CFG, seed=0)
    obs = game.reset(seed=0)
    for _ in range(30):
        decision = agent.act(obs)
        completion = decision.raw_completion

        assert completion.count("ACTION:") == 1
        assert completion.splitlines()[-1] == f"ACTION: {decision.action.id}"
        assert extract_action_token(completion) == decision.action.id

        result = game.step(decision.action)
        obs = result.observation
        if result.terminated or result.truncated:
            break


def test_teacher_records_feed_sft_export(tmp_path: Path):
    from slm_rl.config.schema import TrainConfig
    from slm_rl.datagen.sft_export import export_sft_dataset

    agent, model_id = make_teacher(CFG, seed=0)
    assert model_id == "teacher:freeway_crosser"
    out = tmp_path / "teacher.jsonl"
    with RolloutWriter(out) as writer:
        for i, seed in enumerate(range(3)):
            game = get_game("freeway")(CFG)
            runner = EpisodeRunner(
                game, agent, CFG, writer=writer, model_id=model_id
            )
            runner.run_episode(seed=seed, episode_id=f"t-{i}")

    records = [RolloutRecord.from_json(l) for l in out.read_text().splitlines()]
    assert all(len(r.prompt_messages) == 2 for r in records)
    assert all(r.model_id == "teacher:freeway_crosser" for r in records)

    pairs = export_sft_dataset(out, tmp_path / "sft.jsonl", TrainConfig())
    assert pairs > 0

    exported = (tmp_path / "sft.jsonl").read_text().splitlines()
    assert any("goal" in line for line in exported)


def test_monitor_tolerance_no_repetition_flags():
    # plan 016: a competent Freeway teacher legitimately holds UP for the
    # entire episode (measured: max action streak == max episode length,
    # 682 out of 682). The shipped thresholds (action_repeat_threshold=2000,
    # ngram_loop_threshold=2000, both > max_turns=1500) make repetition
    # detection effectively inert here, by design -- assert it never fires.
    agent, _ = make_teacher(CFG, seed=0)
    clean = 0
    n = 5
    for seed in range(n):
        game = get_game("freeway")(CFG)
        runner = EpisodeRunner(game, agent, CFG)
        summary = runner.run_episode(seed=seed, episode_id=f"m-{seed}")
        signals = summary["monitor"]["signals"]
        assert signals.get("action_repeat", 0) == 0
        assert signals.get("ngram_loop", 0) == 0
        assert signals.get("state_revisit", 0) == 0
        assert set(signals) <= {"reward_stagnation"}
        if summary["monitor"]["interventions"] == 0:
            clean += 1
    assert clean == n
