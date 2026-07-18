"""Boxing via ALE (plan 026 Phase B): registration, RAM decode smoke, menu,
renderer budget, and a short teacher/solver rollout. Skips entirely on
machines without the [atari] extra."""

import pytest

pytest.importorskip("ale_py")
pytest.importorskip("gymnasium")

from slm_rl.config.loader import load_game_config
from slm_rl.datagen.writer import RolloutWriter
from slm_rl.games.atari.ram_maps import boxing as ram_map
from slm_rl.games.registry import get_game
from slm_rl.rollout.runner import EpisodeRunner
from slm_rl.teachers import make_teacher

CFG = load_game_config("boxing")


def test_registration_and_config_load():
    cls = get_game("boxing")
    assert cls.name == "boxing"
    assert CFG.extra["env_id"] == "ALE/Boxing-v5"


def test_ram_decode_smoke():
    game = get_game("boxing")(CFG)
    obs = game.reset(seed=0)
    state = obs.metadata["state"]
    for key in ("player_x", "player_y", "enemy_x", "enemy_y"):
        assert isinstance(state[key], int)
    assert ram_map.decode(game._ram) == state


def test_renderer_budget_and_menu():
    game = get_game("boxing")(CFG)
    obs = game.reset(seed=0)
    assert len(obs.text.splitlines()) <= 12
    assert len(obs.text) < 600
    ids = {a.id for a in obs.legal_actions}
    assert "UPFIRE" in ids
    assert "NOOP" in ids
    assert "opponent" in obs.text.lower()


def test_teacher_short_rollout(tmp_path):
    # Cap max_turns so the test stays a short smoke, not a full 2k-decision bout.
    cfg = CFG.model_copy(update={"max_turns": 40})
    out = tmp_path / "boxing.jsonl"
    with RolloutWriter(out) as writer:
        game = get_game("boxing")(cfg)
        agent, model_id = make_teacher(cfg, seed=0)
        assert model_id == "teacher:boxing_puncher"
        runner = EpisodeRunner(game, agent, cfg, writer=writer)
        summary = runner.run_episode(seed=10_000, episode_id="boxing-0")
        assert isinstance(summary["outcome"], str) and summary["outcome"]
        assert summary["steps"] >= 20
    assert out.read_text().splitlines()
