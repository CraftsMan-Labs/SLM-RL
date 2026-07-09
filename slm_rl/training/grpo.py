"""GRPO strategy: TRL GRPOTrainer + PEFT LoRA on a 4-bit base, per-decision
flattening, KL-to-previous-champion, entropy floor with abort-and-rollback.
Requires the [cuda] extra."""

from __future__ import annotations

from pathlib import Path

from slm_rl.training.base import TrainingStrategy, TrainResult


class GRPOStrategy(TrainingStrategy):
    name = "grpo"

    def train(self, dataset_path: Path, out_dir: Path, init_adapter: Path | None = None) -> TrainResult:
        raise NotImplementedError("Phase 1")
