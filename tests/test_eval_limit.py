"""Regression test for plan 024's eval_episodes fix: GenerationRunner._eval
must pass game_cfg.eval_episodes into run_suite's `limit` (a PREFIX of the
frozen suite seeds, pairing preserved) instead of silently ignoring it and
always playing every suite seed. No real model: FakeBackend, exercises the
REAL _eval -> run_suite -> EpisodeRunner path (unlike test_generation.py,
which monkeypatches _eval itself away)."""

from __future__ import annotations

import pytest

from slm_rl.config.loader import load_run_config
from slm_rl.inference.base import GenOutput, GenParams, InferenceBackend
from slm_rl.rollout import runner as runner_mod
from tiny_game import TinyGame

pytest.importorskip("pyarrow")  # GenerationRunner.__init__ path touches datagen


class FakeBackend(InferenceBackend):
    def generate(self, chats, params: GenParams):
        return [GenOutput(text="ACTION: 1") for _ in chats]

    def load_adapter(self, path):
        pass

    def close(self):
        pass


def _build_runner(tmp_path, monkeypatch, eval_episodes: int):
    import slm_rl.orchestrator.generation as gen

    cfg = load_run_config(game="boxing", overrides={"run_id": "test", "home": str(tmp_path)})
    monkeypatch.setattr(gen, "create_backend", lambda *a, **k: FakeBackend())
    monkeypatch.setattr(gen, "get_game", lambda name: TinyGame)
    runner = gen.GenerationRunner(cfg)
    runner.game_cfg = runner.game_cfg.model_copy(update={"max_turns": 4, "eval_episodes": eval_episodes})
    runner.game_cfg.eval_episodes = eval_episodes
    return runner


def test_eval_episodes_limits_to_a_prefix_of_frozen_seeds(tmp_path, monkeypatch):
    runner = _build_runner(tmp_path, monkeypatch, eval_episodes=7)

    seeds_played = []
    orig_run_episode = runner_mod.EpisodeRunner.run_episode

    def spy_run_episode(self, seed, episode_id):
        seeds_played.append(seed)
        return orig_run_episode(self, seed, episode_id)

    monkeypatch.setattr(runner_mod.EpisodeRunner, "run_episode", spy_run_episode)

    metrics = runner._eval(adapter=None)

    assert metrics["episodes"] == 7
    assert len(seeds_played) == 7
    # first 7 of the suite's frozen seeds (10000..), NOT an arbitrary
    # subset -- pairing across generations requires the SAME prefix every time
    assert seeds_played == list(runner.suite.seeds[:7])


def test_eval_episodes_full_suite_is_unchanged_behavior(tmp_path, monkeypatch):
    # TinyGame suite is 300 seeds; unmodified limit=300 plays every seed.
    runner = _build_runner(tmp_path, monkeypatch, eval_episodes=300)
    metrics = runner._eval(adapter=None)
    assert metrics["episodes"] == 300
    assert metrics["episodes"] == len(runner.suite.seeds)


def test_eval_pruned_limit_not_overridden_by_eval_episodes(tmp_path, monkeypatch):
    # eval_pruned passes its OWN explicit limit (eval_pruned_episodes); the
    # game_cfg.eval_episodes default must not clobber an explicit caller limit.
    runner = _build_runner(tmp_path, monkeypatch, eval_episodes=7)
    metrics = runner._eval(adapter=None, limit=3)
    assert metrics["episodes"] == 3
