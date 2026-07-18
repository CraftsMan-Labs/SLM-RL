"""Bake-pack helpers: URL normalize, MANIFEST validate, local materialize."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_rl.packs import (
    SCHEMA_VERSION,
    cache_slug,
    import_adapter_as_champion,
    materialize_rollouts,
    normalize_repo_id,
    resolve_adapter,
    validate_manifest,
    write_manifest,
)
from slm_rl.orchestrator.registry import ModelRegistry


def test_normalize_and_slug():
    assert normalize_repo_id("org/slm-rl-boxing") == "org/slm-rl-boxing"
    assert (
        normalize_repo_id("https://huggingface.co/datasets/org/slm-rl-boxing")
        == "org/slm-rl-boxing"
    )
    assert (
        normalize_repo_id("https://huggingface.co/BLANK/slm-rl-boxing")
        == "BLANK/slm-rl-boxing"
    )
    assert cache_slug("org/slm-rl-boxing") == "org__slm-rl-boxing"
    with pytest.raises(ValueError, match="empty"):
        normalize_repo_id("  ")
    with pytest.raises(ValueError, match="org/name"):
        normalize_repo_id("nopath")


def test_import_adapter_as_champion(tmp_path: Path):
    src = tmp_path / "src_adapter"
    src.mkdir()
    (src / "adapter_config.json").write_text("{}", encoding="utf-8")
    (src / "adapter_model.safetensors").write_bytes(b"fake")

    run_dir = tmp_path / "run"
    dest = import_adapter_as_champion(
        run_dir, src, model_id="Qwen/Qwen2.5-0.5B", game="boxing",
    )
    assert dest == run_dir / "generations" / "gen_001" / "adapter"
    assert (dest / "adapter_config.json").is_file()
    eval_path = run_dir / "generations" / "gen_001" / "eval" / "results.json"
    assert eval_path.is_file()
    from slm_rl.config.schema import DEFAULT_STUB_PRIMARY

    stub = json.loads(eval_path.read_text())
    assert stub["primary"] == DEFAULT_STUB_PRIMARY
    assert stub["invalid_rate"] == 0.0
    assert stub["intervention_rate"] == 0.0
    reg = ModelRegistry(run_dir / "registry.json")
    assert reg.champion == 1
    assert reg.next_generation == 2


def test_resolve_adapter_uses_cache(tmp_path: Path):
    cached = tmp_path / "packs" / "org__m__adapter" / "adapter"
    cached.mkdir(parents=True)
    (cached / "adapter_config.json").write_text("{}", encoding="utf-8")
    (cached / "adapter_model.safetensors").write_bytes(b"x")
    # Cache hit returns before huggingface_hub import.
    got = resolve_adapter("org/m", tmp_path, "boxing")
    assert got == cached


def test_manifest_game_mismatch():
    m = {"schema_version": SCHEMA_VERSION, "game": "boxing", "n_episodes": 1, "has_dqn": False}
    validate_manifest(m, "boxing")
    with pytest.raises(ValueError, match="freeway"):
        validate_manifest(m, "freeway")
    with pytest.raises(ValueError, match="schema_version"):
        validate_manifest({**m, "schema_version": 999}, "boxing")


def test_materialize_rollouts(tmp_path: Path):
    pack = tmp_path / "pack"
    write_manifest(pack, game="boxing", n_episodes=1, has_dqn=False)
    (pack / "rollouts").mkdir()
    (pack / "rollouts" / "demos.jsonl").write_text(
        json.dumps({"episode_id": "e1", "outcome": "win"}) + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "gen" / "rollouts"
    assert materialize_rollouts(pack, dest) == 1
    assert (dest / "demos.jsonl").is_file()


def test_atari_dqn_loud_miss_all_games(tmp_path: Path, monkeypatch):
    """Missing dqn_checkpoint raises before torch import (all Atari builders)."""
    from slm_rl.config.schema import GameConfig
    from slm_rl.teachers import make_teacher

    # Isolate from a real host `./runs/packs/<game>/dqn.pt` if present.
    monkeypatch.chdir(tmp_path)

    for name in (
        "space-invaders", "freeway", "boxing", "demon-attack",
    ):
        cfg = GameConfig(name=name, max_turns=10)
        with pytest.raises(ValueError, match="dqn_checkpoint not found"):
            make_teacher(cfg, seed=0, dqn_checkpoint="/nonexistent/path/checkpoint.pt")
