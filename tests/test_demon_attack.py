"""Demon Attack via ALE (plan 026 Phase B): registration, RAM decode smoke,
menu, renderer budget, and a short teacher/solver rollout. Skips entirely
on machines without the [atari] extra."""

import pytest

pytest.importorskip("ale_py")
pytest.importorskip("gymnasium")

from slm_rl.config.loader import load_game_config
from slm_rl.datagen.writer import RolloutWriter
from slm_rl.games.atari.ram_maps import demon_attack as ram_map
from slm_rl.games.registry import get_game
from slm_rl.rollout.runner import EpisodeRunner
from slm_rl.teachers import make_teacher

CFG = load_game_config("demon-attack")


def test_registration_and_config_load():
    cls = get_game("demon-attack")
    assert cls.name == "demon-attack"
    assert CFG.extra["env_id"] == "ALE/DemonAttack-v5"


def test_ram_decode_smoke():
    game = get_game("demon-attack")(CFG)
    obs = game.reset(seed=0)
    state = obs.metadata["state"]
    assert isinstance(state["player_x"], int)
    assert isinstance(state["enemy_x"], list) and len(state["enemy_x"]) == 3
    assert isinstance(state["enemy_y"], list) and len(state["enemy_y"]) == 3
    assert isinstance(state["missile_in_flight"], bool)
    # AtariARI num_lives did not verify -- must not appear in decode()
    assert "num_lives" not in state
    assert ram_map.decode(game._ram) == state


def test_renderer_budget_and_menu():
    game = get_game("demon-attack")(CFG)
    obs = game.reset(seed=0)
    assert len(obs.text.splitlines()) <= 12
    assert len(obs.text) < 600
    ids = {a.id for a in obs.legal_actions}
    assert ids == {"NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"}
    assert "demon" in obs.text.lower() or "ship" in obs.text.lower()


def test_teacher_short_rollout(tmp_path):
    cfg = CFG.model_copy(update={"max_turns": 40})
    out = tmp_path / "demon-attack.jsonl"
    with RolloutWriter(out) as writer:
        game = get_game("demon-attack")(cfg)
        agent, model_id = make_teacher(cfg, seed=0)
        assert model_id == "teacher:demon_attack_tracker"
        runner = EpisodeRunner(game, agent, cfg, writer=writer)
        summary = runner.run_episode(seed=10_000, episode_id="demon-0")
        assert isinstance(summary["outcome"], str) and summary["outcome"]
        assert summary["steps"] >= 20
    assert out.read_text().splitlines()
