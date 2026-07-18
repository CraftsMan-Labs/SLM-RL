"""GRPO workshop budget: max_steps / epochs / group_size wiring (no real train)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

from slm_rl.config.schema import GameConfig, TrainConfig
from slm_rl.training.grpo import GRPOStrategy


def _install_fake_train_stack(monkeypatch, captured: dict) -> None:
    """Stub torch / datasets / trl so GRPOStrategy.train runs offline."""
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", torch)

    datasets = ModuleType("datasets")

    def load_dataset(*_a, **_k):
        return [{"prompt": [], "game_ctx": "{}"}]

    datasets.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets)

    class FakeGRPOConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeTrainer:
        def __init__(self, **_kwargs):
            self.state = SimpleNamespace(log_history=[])
            self.model = MagicMock()

        def train(self):
            return None

    trl = ModuleType("trl")
    trl.GRPOConfig = FakeGRPOConfig
    trl.GRPOTrainer = FakeTrainer
    monkeypatch.setitem(sys.modules, "trl", trl)

    monkeypatch.setattr(
        "slm_rl.training.grpo.bootstrap_lora",
        lambda *_a, **_k: (MagicMock(), MagicMock()),
    )
    monkeypatch.setattr("slm_rl.training.grpo.release_trainer_memory", lambda *_a, **_k: None)


def test_cpu_grpo_honors_max_steps_and_group_cap(tmp_path, monkeypatch):
    captured: dict = {}
    _install_fake_train_stack(monkeypatch, captured)

    src = tmp_path / "train.jsonl"
    src.write_text(
        json.dumps(
            {
                "episode_id": "e1",
                "step_idx": 0,
                "seed": 0,
                "parsed_action": "UP",
                "reward": 1.0,
                "generation": 1,
                "legal_actions": [{"id": "UP"}],
                "prompt_messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "board"},
                ],
                "parse_status": "ok",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = TrainConfig(
        group_size=8,
        grpo_max_steps=24,
        grpo_max_prompts=32,
        max_completion_tokens=24,
    )
    strategy = GRPOStrategy(cfg, "LiquidAI/LFM2.5-350M", GameConfig(name="boxing"))
    result = strategy.train(src, tmp_path / "out")

    assert captured["max_steps"] == 24
    assert captured["num_train_epochs"] == 1
    assert captured["num_generations"] == 2  # CPU hard cap
    assert (
        captured["per_device_train_batch_size"] * captured["gradient_accumulation_steps"]
    ) % captured["num_generations"] == 0
    assert result.metrics["num_prompts"] == 1
    assert (tmp_path / "grpo.jsonl").exists()


def test_cpu_grpo_unlimited_steps_when_unset(tmp_path, monkeypatch):
    captured: dict = {}
    _install_fake_train_stack(monkeypatch, captured)

    src = tmp_path / "train.jsonl"
    src.write_text(
        json.dumps(
            {
                "episode_id": "e1",
                "step_idx": 0,
                "seed": 0,
                "parsed_action": "UP",
                "reward": 1.0,
                "generation": 1,
                "legal_actions": [{"id": "UP"}],
                "prompt_messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "board"},
                ],
                "parse_status": "ok",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = TrainConfig(group_size=2, grpo_max_steps=None, grpo_max_prompts=8)
    strategy = GRPOStrategy(cfg, "m", GameConfig(name="boxing"))
    strategy.train(src, tmp_path / "out")
    assert captured["max_steps"] == -1
    assert captured["num_train_epochs"] == 1
