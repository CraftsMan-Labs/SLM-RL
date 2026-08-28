"""CPU-safe checks for the optional Mario DQN workshop helper."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKSHOP = ROOT / "docs" / "workshop"
sys.path.insert(0, str(WORKSHOP))

from mario_lab import (  # noqa: E402
    CHECKPOINT_N_ACTIONS,
    CLIP_FILES,
    DEFAULT_EVAL_STEPS,
    DEFAULT_MARIO_MODEL_REPO,
    EVAL_STEPS_RANGE,
    HF_CHECKPOINTS,
    LOCAL_TRAINED_NAME,
    STAGED_CHECKPOINTS,
    TWO_ACTIONS,
    download_hf_checkpoint,
    epsilon_at,
    evaluate_mario,
    extract_online_state,
    fallback_paths,
    hf_checkpoint_spec,
    load_clip_manifest,
    load_fallback_metrics,
    load_training_checkpoint,
    make_qnet,
    pinned_packages,
    preprocess_frame,
    record_mario_clips,
    remap_online_keys,
    resolve_named_checkpoint,
    run_mario_demo,
    save_training_checkpoint,
    sha256_file,
    stack_frames,
    train_mario_live,
    verify_checkpoint,
)


def test_pinned_packages_are_versioned():
    pkgs = pinned_packages()
    assert pkgs
    assert any("gym-super-mario-bros==" in p for p in pkgs)


def test_preprocess_and_stack():
    rgb = np.zeros((240, 256, 3), dtype=np.uint8)
    rgb[10, 20] = (255, 0, 0)
    frame = preprocess_frame(rgb)
    assert frame.shape == (84, 84)
    assert frame.max() <= 1.0
    stacked = stack_frames([frame])
    assert stacked.shape == (4, 84, 84)


def test_fallback_assets_exist():
    paths = fallback_paths()
    assert paths["storyboard"].is_file()
    assert paths["metrics"].is_file()
    rows = load_fallback_metrics()
    assert rows[0]["stage"] == "untrained"
    assert rows[-1]["stage"] == "pretrained"


def test_checkpoint_verify(tmp_path):
    missing = tmp_path / "nope.pt"
    assert verify_checkpoint(missing, "abcd") is False
    blob = tmp_path / "weights.pt"
    blob.write_bytes(b"hello")
    digest = sha256_file(blob)
    assert verify_checkpoint(blob, digest) is True
    assert verify_checkpoint(blob, "0" * 64) is False


def test_run_demo_falls_back_without_env(tmp_path, monkeypatch):
    import mario_lab

    monkeypatch.setattr(mario_lab, "try_make_mario_env", lambda: (None, "import failed: test"))
    result = run_mario_demo(tmp_path, play_steps=10, continue_steps=0)
    assert result["mode"] == "fallback"
    assert "import failed" in result["reason"]
    assert result["fallback_metrics"]
    assert result["encoder"] == "pixels → CNN"
    assert result["teacher_encoder"] == "RAM vector → MLP"


def test_qnet_shape_if_torch():
    torch = pytest.importorskip("torch")
    net = make_qnet(CHECKPOINT_N_ACTIONS)
    out = net(torch.zeros(2, 4, 84, 84))
    assert tuple(out.shape) == (2, CHECKPOINT_N_ACTIONS)


def test_staged_checkpoints_are_pinned():
    stages = [row["stage"] for row in STAGED_CHECKPOINTS]
    assert stages == ["untrained", "mid", "pretrained"]
    assert TWO_ACTIONS == [["right"], ["right", "A"]]
    for row in STAGED_CHECKPOINTS:
        assert row["clip"] in CLIP_FILES
        if row["url"]:
            assert len(row["sha256"]) == 64
            assert row["filename"]


def test_remap_public_mario_keys():
    raw = {
        "model": {
            "online.0.weight": "c0",
            "online.2.weight": "c2",
            "online.4.weight": "c4",
            "online.7.weight": "h1",
            "online.9.weight": "h3",
            "target.0.weight": "ignore",
        }
    }
    state = extract_online_state(raw)
    assert state["conv.0.weight"] == "c0"
    assert state["conv.2.weight"] == "c2"
    assert state["head.1.weight"] == "h1"
    assert state["head.3.weight"] == "h3"
    assert "target.0.weight" not in state
    already = remap_online_keys({"conv.0.weight": 1, "head.3.bias": 2})
    assert already["conv.0.weight"] == 1


def test_record_clips_falls_back_without_env(tmp_path, monkeypatch):
    import mario_lab

    monkeypatch.setattr(mario_lab, "try_make_mario_env", lambda: (None, "import failed: test"))
    result = record_mario_clips(tmp_path, workdir=tmp_path / "ckpts")
    assert result["mode"] == "fallback"
    assert "import failed" in result["reason"]


def test_recorded_clips_and_manifest_exist():
    paths = fallback_paths()
    for name in CLIP_FILES:
        clip = paths[Path(name).stem]
        assert clip.is_file()
        assert clip.stat().st_size > 1000
    manifest = load_clip_manifest()
    assert manifest.get("metrics")
    stages = [row["stage"] for row in manifest["metrics"]]
    assert stages[0] == "untrained"
    assert stages[-1] == "pretrained"


def test_hf_repo_template_exists():
    root = WORKSHOP / "assets" / "mario" / "hf-repo"
    for name in ("README.md", "config.json", "checksums.json", "metrics.json"):
        assert (root / name).is_file()
    card = (root / "README.md").read_text(encoding="utf-8")
    assert "Nintendo" in card
    assert "warm-start.chkpt" in card
    assert "final.chkpt" in card


def test_hf_checkpoint_catalog_is_pinned():
    assert DEFAULT_MARIO_MODEL_REPO.endswith("mario-dqn-workshop")
    assert set(HF_CHECKPOINTS) == {"warm-start", "final"}
    for name, row in HF_CHECKPOINTS.items():
        spec = hf_checkpoint_spec(name)
        assert spec["filename"].endswith(".chkpt")
        assert len(spec["sha256"]) == 64
        assert spec["fallback_stage"] in {r["stage"] for r in STAGED_CHECKPOINTS}


def test_download_hf_rejects_bad_checksum(tmp_path, monkeypatch):
    blob = tmp_path / "cached.chkpt"
    blob.write_bytes(b"wrong")

    def fake_download(**kwargs):
        assert kwargs["repo_id"] == "demo/mario-dqn-workshop"
        assert kwargs["revision"] == "abc123"
        assert kwargs["filename"] == "warm-start.chkpt"
        assert kwargs["token"] is False
        return str(blob)

    monkeypatch.setitem(sys.modules, "huggingface_hub", type("Hub", (), {"hf_hub_download": staticmethod(fake_download)})())
    dest = tmp_path / "out.chkpt"
    assert download_hf_checkpoint(
        dest,
        repo_id="demo/mario-dqn-workshop",
        filename="warm-start.chkpt",
        revision="abc123",
        sha256="0" * 64,
    ) is None
    assert not dest.is_file()


def test_download_hf_copies_verified_file(tmp_path, monkeypatch):
    blob = tmp_path / "cached.chkpt"
    blob.write_bytes(b"weights")
    digest = sha256_file(blob)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        type("Hub", (), {"hf_hub_download": staticmethod(lambda **_k: str(blob))})(),
    )
    dest = tmp_path / "out.chkpt"
    got = download_hf_checkpoint(
        dest,
        repo_id="demo/mario-dqn-workshop",
        filename="final.chkpt",
        revision="main",
        sha256=digest,
    )
    assert got == dest
    assert dest.read_bytes() == b"weights"


def test_resolve_named_checkpoint_falls_back_to_github(tmp_path, monkeypatch):
    import mario_lab

    monkeypatch.setattr(mario_lab, "download_hf_checkpoint", lambda *_a, **_k: None)
    fallback = tmp_path / "mid.chkpt"
    fallback.write_bytes(b"mid")
    monkeypatch.setattr(mario_lab, "resolve_staged_checkpoint", lambda *_a, **_k: fallback)
    path, source = resolve_named_checkpoint("warm-start", tmp_path)
    assert path == fallback
    assert source.startswith("github-fallback:")


def test_resolve_local_trained_missing(tmp_path):
    path, source = resolve_named_checkpoint("local-trained", tmp_path)
    assert path is None
    assert "missing" in source


def test_train_and_eval_fall_back_without_env(tmp_path, monkeypatch):
    import mario_lab

    monkeypatch.setattr(mario_lab, "try_make_mario_env", lambda: (None, "import failed: test"))
    trained = train_mario_live(tmp_path, train_minutes=1, eval_interval=50)
    assert trained["mode"] == "fallback"
    assert "import failed" in trained["reason"]
    assert trained["fallback_metrics"]
    evaluated = evaluate_mario(tmp_path, eval_source="public-final", eval_steps=200)
    assert evaluated["mode"] == "fallback"
    assert evaluated["eval_steps"] == 200


def test_evaluate_clamps_step_budget(tmp_path, monkeypatch):
    import mario_lab

    monkeypatch.setattr(mario_lab, "try_make_mario_env", lambda: (None, "import failed: test"))
    high = evaluate_mario(tmp_path, eval_steps=9_999_999)
    assert high["eval_steps"] == EVAL_STEPS_RANGE[1]
    low = evaluate_mario(tmp_path, eval_steps=1)
    assert low["eval_steps"] == EVAL_STEPS_RANGE[0]
    assert DEFAULT_EVAL_STEPS == 10_000


class _FakeMarioEnv:
    def reset(self, *args, **kwargs):
        return np.zeros((24, 32, 3), dtype=np.uint8)

    def step(self, action):
        return np.zeros((24, 32, 3), dtype=np.uint8), 0.1, False, {"x_pos": 12}

    def close(self):
        pass


def test_chunked_train_saves_and_resumes_if_torch(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import mario_lab

    monkeypatch.setattr(mario_lab, "try_make_mario_env", lambda: (_FakeMarioEnv(), ""))
    monkeypatch.setattr(mario_lab, "MIN_REPLAY", 8)
    monkeypatch.setattr(mario_lab, "BATCH_SIZE", 8)
    monkeypatch.setattr(mario_lab, "TARGET_SYNC_EVERY", 4)
    rows: list[dict] = []
    first = train_mario_live(
        tmp_path,
        training_mode="from-scratch",
        train_minutes=1,
        eval_interval=8,
        seed=0,
        on_progress=rows.append,
        max_decisions=16,
        eval_decisions=4,
    )
    if first["mode"] != "live":
        pytest.skip(f"live train unavailable: {first['reason']}")
    assert first["history"]
    assert rows
    assert (tmp_path / LOCAL_TRAINED_NAME).is_file()
    q_net = make_qnet(CHECKPOINT_N_ACTIONS)
    meta = load_training_checkpoint(q_net, tmp_path / LOCAL_TRAINED_NAME)
    assert meta["decisions"] == first["decisions"]
    second = train_mario_live(
        tmp_path,
        training_mode="from-scratch",
        train_minutes=1,
        eval_interval=8,
        seed=1,
        max_decisions=first["decisions"] + 8,
        eval_decisions=4,
    )
    assert second["mode"] == "live"
    assert second["source"] == "local-resume"
    assert second["decisions"] >= first["decisions"]
    assert epsilon_at(0) == pytest.approx(1.0)
    _ = torch


def test_evaluate_aggregates_metrics_if_torch(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import mario_lab

    monkeypatch.setattr(mario_lab, "try_make_mario_env", lambda: (_FakeMarioEnv(), ""))
    monkeypatch.setattr(mario_lab, "encode_mp4", lambda *_a, **_k: False)
    q_net = make_qnet(CHECKPOINT_N_ACTIONS)
    target = make_qnet(CHECKPOINT_N_ACTIONS)
    opt = __import__("torch").optim.Adam(q_net.parameters(), lr=1e-4)
    save_training_checkpoint(
        tmp_path / LOCAL_TRAINED_NAME,
        q_net,
        target,
        opt,
        {"decisions": 10, "epsilon": 0.9, "training_mode": "from-scratch"},
    )
    result = evaluate_mario(tmp_path, eval_source="local-trained", eval_steps=120, collect_frames=False)
    if result["mode"] != "live":
        pytest.skip(f"evaluate unavailable: {result['reason']}")
    assert result["eval_source"] == "local-trained"
    assert result["completed_episodes"] >= 1
    assert "total_reward" in result
    assert "farthest_distance" in result
    assert "deaths" in result
    assert "best_attempt" in result
