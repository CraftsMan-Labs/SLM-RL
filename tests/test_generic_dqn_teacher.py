"""Generic DQN teacher fallback for plugin games (vector_obs + checkpoint)."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_rl.config.schema import GameConfig
from slm_rl.games.registry import _REGISTRY
from slm_rl.teachers import make_teacher
from tests.tiny_game import TinyGame


@pytest.fixture
def tiny_registered():
    """Register TinyGame under a unique name for the duration of a test."""
    name = "tiny-plugin-dqn"
    if name not in _REGISTRY:
        TinyGame.name = name
        _REGISTRY[name] = TinyGame
    try:
        yield name
    finally:
        _REGISTRY.pop(name, None)


def test_make_teacher_generic_dqn_with_checkpoint(tmp_path: Path, monkeypatch, tiny_registered):
    ckpt = tmp_path / "dqn.pt"
    ckpt.write_bytes(b"fake")

    class _Stub:
        model_id = "teacher:tiny_plugin_dqn_dqn"

        def __init__(self, checkpoint, system_prompt, seed=None):
            self.checkpoint = checkpoint

    monkeypatch.setattr("slm_rl.teachers.dqn.DQNTeacherAgent", _Stub)
    import slm_rl.teachers.dqn as dqn_mod

    monkeypatch.setattr(dqn_mod, "DQNTeacherAgent", _Stub)

    cfg = GameConfig(name=tiny_registered, max_turns=10)
    agent, model_id = make_teacher(cfg, seed=0, dqn_checkpoint=str(ckpt))
    assert model_id == "teacher:tiny_plugin_dqn_dqn"
    assert Path(agent.checkpoint) == ckpt


def test_make_teacher_plugin_without_checkpoint_raises(tiny_registered):
    cfg = GameConfig(name=tiny_registered, max_turns=10)
    with pytest.raises(ValueError, match="No teacher implemented"):
        make_teacher(cfg, seed=0)


def test_make_teacher_plugin_missing_checkpoint_raises(tmp_path: Path, monkeypatch, tiny_registered):
    monkeypatch.chdir(tmp_path)
    cfg = GameConfig(name=tiny_registered, max_turns=10)
    with pytest.raises(ValueError, match="dqn_checkpoint not found"):
        make_teacher(cfg, seed=0, dqn_checkpoint="/nonexistent/path/checkpoint.pt")


def test_make_teacher_plugin_resolves_canonical_path(tmp_path: Path, monkeypatch, tiny_registered):
    monkeypatch.chdir(tmp_path)
    pack = tmp_path / "runs" / "teachers" / f"dqn-{tiny_registered}.pt"
    pack.parent.mkdir(parents=True)
    pack.write_bytes(b"fake")

    class _Stub:
        model_id = "teacher:resolved"

        def __init__(self, checkpoint, system_prompt, seed=None):
            self.checkpoint = checkpoint

    monkeypatch.setattr("slm_rl.teachers.dqn.DQNTeacherAgent", _Stub)
    import slm_rl.teachers.dqn as dqn_mod

    monkeypatch.setattr(dqn_mod, "DQNTeacherAgent", _Stub)

    cfg = GameConfig(name=tiny_registered, max_turns=10)
    agent, model_id = make_teacher(
        cfg, seed=0, dqn_checkpoint="/nonexistent/path/checkpoint.pt",
    )
    assert model_id == "teacher:resolved"
    assert Path(agent.checkpoint) == pack.resolve()
