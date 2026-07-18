"""Exhibition runner (plan 020): base-vs-champion replay written under
synthetic `theater/<side>/` run dirs. FakeBackend only -- no GPU, no model,
per CODING_GUIDELINE. Reuses the FakeBackend/create_backend-monkeypatch
pattern from tests/test_generation.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import slm_rl.theater.exhibition as exhibition_mod
from slm_rl.inference.base import GenOutput, GenParams, InferenceBackend
from slm_rl.orchestrator.paths import RunPaths
from slm_rl.orchestrator.registry import ModelRegistry
from tiny_game import TinyGame


class FakeBackend(InferenceBackend):
    """Always emits a legal, parseable move; records adapter loads/closes so
    tests can assert the 8GB one-model-at-a-time handoff."""

    instances: list["FakeBackend"] = []

    def __init__(self, model_id: str, quantization=None):
        self.model_id = model_id
        self.adapter_path = None
        self.closed = False
        FakeBackend.instances.append(self)

    def generate(self, chats, params: GenParams):
        return [GenOutput(text="ACTION: 1")]

    def load_adapter(self, path):
        self.adapter_path = path

    def close(self):
        self.closed = True


def _make_run(tmp_path: Path, *, champion: int, game: str = "boxing") -> Path:
    """A minimal `runs/<run_id>/` skeleton: run_config.yaml + registry.json.
    No actual generations/ needed -- the exhibition runner reads only these
    two files to resolve model/backend/champion."""
    run_id = "theater-test"
    home = tmp_path / "runs"
    paths = RunPaths(home, run_id)
    paths.root.mkdir(parents=True)

    import yaml

    (paths.root / "run_config.yaml").write_text(
        yaml.safe_dump({
            "run_id": run_id, "home": str(home), "game": game,
            "model": "fake/model", "backend": "transformers", "tier": None,
        }),
        encoding="utf-8",
    )

    registry = ModelRegistry(paths.registry)
    if champion > 0:
        registry._data["champion"] = champion
        registry._data["history"].append(
            {"generation": champion, "event": "promoted", "reason": "test"}
        )
        registry._save()
        adapter_dir = paths.adapter(champion)
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "adapter_model.safetensors").write_text("weights")
    return paths.root


@pytest.fixture(autouse=True)
def _patch_backend(monkeypatch):
    import slm_rl.games.registry as registry

    FakeBackend.instances = []
    monkeypatch.setattr(
        exhibition_mod, "create_backend",
        lambda name, model_id, quantization=None: FakeBackend(model_id, quantization),
    )
    # ponytail: TinyGame stand-in — no ALE for unit tests
    monkeypatch.setattr(registry, "get_game", lambda name: TinyGame)
    yield


def test_exhibition_writes_both_sides_in_viewer_layout(tmp_path):
    run_dir = _make_run(tmp_path, champion=2)

    result = exhibition_mod.run_exhibition(run_dir, "boxing", episodes=3)

    assert result.champion_dir is not None
    assert result.champion_generation == 2
    assert result.message is None

    base_rollout = run_dir / "theater" / "base" / "generations" / "gen_000" / "rollouts" / "boxing.jsonl"
    champ_rollout = run_dir / "theater" / "champion" / "generations" / "gen_002" / "rollouts" / "boxing.jsonl"
    assert base_rollout.exists()
    assert champ_rollout.exists()

    base_lines = [json.loads(l) for l in base_rollout.read_text().splitlines()]
    champ_lines = [json.loads(l) for l in champ_rollout.read_text().splitlines()]
    assert {r["episode_id"] for r in base_lines}  # non-empty
    assert all(r["generation"] == 0 for r in base_lines)
    assert all(r["generation"] == 2 for r in champ_lines)
    assert all(r["model_id"] == "fake/model" for r in base_lines + champ_lines)
    assert all(r["adapter_ref"] is None for r in base_lines)
    assert all(r["adapter_ref"] is not None for r in champ_lines)


def test_exhibition_seeds_start_at_20000_and_match_across_sides(tmp_path):
    run_dir = _make_run(tmp_path, champion=1)
    exhibition_mod.run_exhibition(run_dir, "boxing", episodes=3)

    base_rollout = run_dir / "theater" / "base" / "generations" / "gen_000" / "rollouts" / "boxing.jsonl"
    champ_rollout = run_dir / "theater" / "champion" / "generations" / "gen_001" / "rollouts" / "boxing.jsonl"

    base_seeds = sorted({json.loads(l)["seed"] for l in base_rollout.read_text().splitlines()})
    champ_seeds = sorted({json.loads(l)["seed"] for l in champ_rollout.read_text().splitlines()})

    assert base_seeds == [20_000, 20_001, 20_002]
    assert champ_seeds == base_seeds

    from slm_rl.config.loader import load_game_config

    game_cfg = load_game_config("boxing")
    assert min(base_seeds) >= 20_000 > game_cfg.eval_seeds_start


def test_exhibition_one_backend_at_a_time(tmp_path):
    run_dir = _make_run(tmp_path, champion=1)
    exhibition_mod.run_exhibition(run_dir, "boxing", episodes=2)

    assert len(FakeBackend.instances) == 2
    base_be, champ_be = FakeBackend.instances
    assert base_be.closed is True
    assert champ_be.closed is True
    assert base_be.adapter_path is None
    assert champ_be.adapter_path is not None


def test_exhibition_champion_less_run_is_base_only_with_message(tmp_path):
    run_dir = _make_run(tmp_path, champion=0)
    result = exhibition_mod.run_exhibition(run_dir, "boxing", episodes=2)

    assert result.champion_dir is None
    assert result.champion_generation == 0
    assert "no promoted champion" in result.message

    base_rollout = run_dir / "theater" / "base" / "generations" / "gen_000" / "rollouts" / "boxing.jsonl"
    assert base_rollout.exists()
    assert not (run_dir / "theater" / "champion").exists()
    assert len(FakeBackend.instances) == 1
    status = json.loads((run_dir / "theater" / "status.json").read_text(encoding="utf-8"))
    assert status["phase"] == "done"
    assert "no promoted champion" in status["message"]


def test_exhibition_writes_status_through_champion(tmp_path):
    run_dir = _make_run(tmp_path, champion=1)
    exhibition_mod.run_exhibition(run_dir, "boxing", episodes=2)
    status = json.loads((run_dir / "theater" / "status.json").read_text(encoding="utf-8"))
    assert status["phase"] == "done"
    assert status["side"] == "champion"
    assert status["completed"] == 2


def test_exhibition_rerun_overwrites_not_appends(tmp_path):
    run_dir = _make_run(tmp_path, champion=1)
    exhibition_mod.run_exhibition(run_dir, "boxing", episodes=2)
    exhibition_mod.run_exhibition(run_dir, "boxing", episodes=2)

    base_rollout = run_dir / "theater" / "base" / "generations" / "gen_000" / "rollouts" / "boxing.jsonl"
    lines = base_rollout.read_text().splitlines()
    episode_ids = {json.loads(l)["episode_id"] for l in lines}
    assert len(episode_ids) == 2


def test_exhibition_viewer_reuse_iter_run_records_and_replay_frames(tmp_path, monkeypatch):
    """Design decisions point 1: a theater/<side>/ dir IS a valid viewer
    run_dir by construction -- iter_run_records must read it with zero
    changes."""
    import threading

    from slm_rl.config.schema import GameConfig
    from slm_rl.webui.tailer import iter_run_records

    run_dir = _make_run(tmp_path, champion=1)
    exhibition_mod.run_exhibition(run_dir, "boxing", episodes=2)

    base_dir = run_dir / "theater" / "base"
    stop = threading.Event()
    stop.set()
    records = list(iter_run_records(base_dir, stop=stop))
    assert records
    # EpisodeRunner stamps config.name (boxing); file is boxing.jsonl
    assert all(r["game"] == "boxing" for r in records)

    # ReplayUnavailable when the game config has no env_id
    from slm_rl.webui.replay import ReplayUnavailable, replay_frames

    monkeypatch.setattr(
        "slm_rl.webui.replay.load_game_config",
        lambda g: GameConfig(name=g, max_turns=10, extra={}),
    )
    first_episode = records[0]["episode_id"]
    with pytest.raises(ReplayUnavailable):
        next(replay_frames(base_dir, first_episode))


def test_mark_theater_ui_stopped_stamps_failed_phase(tmp_path: Path):
    theater = tmp_path / "theater"
    theater.mkdir()
    (theater / "status.json").write_text(
        json.dumps({"phase": "base", "episode": 1, "episodes": 4}) + "\n",
        encoding="utf-8",
    )
    exhibition_mod.mark_theater_ui_stopped(theater)
    data = json.loads((theater / "status.json").read_text(encoding="utf-8"))
    assert data["phase"] == "failed"
    assert data["error"] == "stopped via UI"
    # Missing theater/ must not raise.
    exhibition_mod.mark_theater_ui_stopped(tmp_path / "no-such-theater")
