"""Doom-loop machinery + end-to-end random rollouts (the CPU smoke path)."""

import json
from pathlib import Path

from slm_rl.agents.base import ActionDecision, Agent
from slm_rl.agents.bots import RandomAgent
from slm_rl.config.loader import load_game_config
from slm_rl.datagen.schema import RolloutRecord
from slm_rl.datagen.writer import RolloutWriter
from slm_rl.eval.gate import EvalGate
from slm_rl.config.schema import GateConfig
from slm_rl.games.mastermind import MastermindGame
from slm_rl.rollout.runner import EpisodeRunner

CFG = load_game_config("mastermind")


class DegenerateAgent(Agent):
    """Always plays the same move — must trigger the intervention ladder."""

    def act(self, obs, history) -> ActionDecision:
        action = obs.legal_actions[0]
        return ActionDecision(action=action, raw_completion=f"ACTION: {action.id}")


def test_degenerate_agent_triggers_reflect_then_truncate():
    runner = EpisodeRunner(MastermindGame(CFG), DegenerateAgent(), CFG)
    summary = runner.run_episode(seed=11, episode_id="doom")
    kinds = summary["monitor"]["intervention_kinds"]
    # mastermind config: action_repeat_threshold=2, ladder [reflect, truncate]
    assert kinds[0] == "reflect"
    assert kinds[-1] == "truncate"
    assert summary["outcome"] == "truncated"
    assert summary["steps"] < CFG.max_turns  # ended early, not by turn cap


def test_random_rollout_writes_valid_records(tmp_path: Path):
    out = tmp_path / "rollout.jsonl"
    with RolloutWriter(out) as writer:
        for i in range(3):
            runner = EpisodeRunner(
                MastermindGame(CFG),
                RandomAgent(seed=i),
                CFG,
                writer=writer,
                run_id="test",
                model_id="random",
            )
            summary = runner.run_episode(seed=i, episode_id=f"ep-{i}")
            assert summary["outcome"] in ("win", "loss", "truncated")

    lines = out.read_text().strip().splitlines()
    assert len(lines) >= 3
    records = [RolloutRecord.from_json(line) for line in lines]
    # exactly one terminal record per episode, cum_reward monotone bookkeeping
    terminal = [r for r in records if r.terminated or r.truncated]
    assert len(terminal) == 3
    assert all(r.outcome is not None for r in terminal)
    assert all(r.game == "mastermind" and r.run_id == "test" for r in records)
    step0 = [r for r in records if r.step_idx == 0]
    assert len(step0) == 3


def test_consolidate_to_parquet(tmp_path: Path):
    pa = __import__("pytest").importorskip("pyarrow")  # noqa: F841
    import pyarrow.parquet as pq

    from slm_rl.datagen.consolidate import consolidate

    with RolloutWriter(tmp_path / "r.jsonl") as writer:
        runner = EpisodeRunner(
            MastermindGame(CFG), RandomAgent(seed=0), CFG, writer=writer
        )
        runner.run_episode(seed=0, episode_id="ep-0")

    out = tmp_path / "train.parquet"
    rows = consolidate(tmp_path, out, chunk_rows=2)
    table = pq.read_table(out)
    assert table.num_rows == rows > 0
    first = table.to_pylist()[0]
    assert json.loads(first["prompt_messages"]) == []  # RandomAgent has no prompt


def test_eval_gate_promote_and_reject():
    gate = EvalGate(GateConfig())
    champ = {"primary": 0.30, "invalid_rate": 0.01, "intervention_rate": 0.05, "mean_entropy": 0.5}

    better = {"primary": 0.40, "invalid_rate": 0.02, "intervention_rate": 0.05, "mean_entropy": 0.5}
    promote, reason = gate.decide(champ, better)
    assert promote, reason

    no_gain = dict(better, primary=0.31)
    assert not gate.decide(champ, no_gain)[0]

    loopy = dict(better, intervention_rate=0.20)
    assert not gate.decide(champ, loopy)[0]

    invalid = dict(better, invalid_rate=0.50)
    assert not gate.decide(champ, invalid)[0]

    collapsed = dict(better, mean_entropy=0.05)
    assert not gate.decide(champ, collapsed)[0]

    unknown_entropy = dict(better, mean_entropy=None)
    assert gate.decide(champ, unknown_entropy)[0]  # entropy check skipped
