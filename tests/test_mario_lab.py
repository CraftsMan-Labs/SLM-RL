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
    STAGED_CHECKPOINTS,
    TWO_ACTIONS,
    extract_online_state,
    fallback_paths,
    load_clip_manifest,
    load_fallback_metrics,
    make_qnet,
    pinned_packages,
    preprocess_frame,
    record_mario_clips,
    remap_online_keys,
    run_mario_demo,
    sha256_file,
    stack_frames,
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
