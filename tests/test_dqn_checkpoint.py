"""Per-game DQN checkpoint resolution (bake pack + train-dqn paths)."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_rl.config.schema import GameConfig
from slm_rl.teachers import make_teacher
from slm_rl.teachers.dqn_checkpoint import find_dqn_checkpoint, missing_dqn_hint


def test_find_dqn_checkpoint_prefers_pack(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pack = tmp_path / "runs" / "packs" / "boxing" / "dqn.pt"
    pack.parent.mkdir(parents=True)
    pack.write_bytes(b"x")
    teachers = tmp_path / "runs" / "teachers" / "dqn-boxing.pt"
    teachers.parent.mkdir(parents=True)
    teachers.write_bytes(b"y")
    found = find_dqn_checkpoint("boxing", tmp_path / "runs")
    assert found == pack.resolve()


def test_make_teacher_resolves_wrong_legacy_path(tmp_path: Path, monkeypatch):
    """Configured space-invaders path missing → use this game's pack."""
    monkeypatch.chdir(tmp_path)
    pack = tmp_path / "runs" / "packs" / "demon-attack" / "dqn.pt"
    pack.parent.mkdir(parents=True)
    pack.write_bytes(b"fake")

    # Avoid loading a real DQN net — only assert we don't raise on path miss
    # before the torch import by stubbing DQNTeacherAgent.
    class _Stub:
        model_id = "teacher:demon_attack_dqn"

        def __init__(self, checkpoint, system_prompt, seed=None):
            self.checkpoint = checkpoint

    monkeypatch.setattr("slm_rl.teachers.dqn.DQNTeacherAgent", _Stub)
    # Import path used inside _maybe_dqn after the file check
    import slm_rl.teachers.dqn as dqn_mod

    monkeypatch.setattr(dqn_mod, "DQNTeacherAgent", _Stub)

    cfg = GameConfig(name="demon-attack", max_turns=10)
    agent, model_id = make_teacher(
        cfg, seed=0, dqn_checkpoint="runs/teachers/dqn-space-invaders.pt",
    )
    assert model_id == "teacher:demon_attack_dqn"
    assert Path(agent.checkpoint) == pack.resolve()


def test_missing_hint_names_game():
    assert "boxing" in missing_dqn_hint("boxing")
    assert "dqn-boxing.pt" in missing_dqn_hint("boxing")


def test_loud_miss_still_raises_when_nothing_on_disk(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = GameConfig(name="boxing", max_turns=10)
    with pytest.raises(ValueError, match="No DQN checkpoint for 'boxing'"):
        make_teacher(cfg, seed=0, dqn_checkpoint="/nonexistent/path/checkpoint.pt")


def test_legacy_si_default_falls_back_to_heuristic(tmp_path: Path, monkeypatch):
    """Old playground wrote dqn-space-invaders.pt for every game."""
    monkeypatch.chdir(tmp_path)
    cfg = GameConfig(name="boxing", max_turns=10)
    agent, model_id = make_teacher(
        cfg, seed=0, dqn_checkpoint="runs/teachers/dqn-space-invaders.pt",
    )
    assert model_id == "teacher:boxing_puncher"
    assert agent is not None
