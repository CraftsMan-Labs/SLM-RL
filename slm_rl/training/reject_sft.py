"""Rejection-sampling SFT (STaR/ReST-style) — the universal 8GB training
path, and the gen-0 warm-start for GRPO tiers.

Select winning / top-quantile, monitor-clean trajectories (datagen/
sft_export.py, with a diversity quota) then LoRA-SFT on the (prompt ->
winning action) pairs: mlx-lm on Apple Silicon, transformers+PEFT elsewhere
(CPU works, slowly)."""

from __future__ import annotations

from pathlib import Path

from slm_rl.training.base import TrainingStrategy, TrainResult


class RejectSFTStrategy(TrainingStrategy):
    name = "reject_sft"

    def train(self, dataset_path: Path, out_dir: Path, init_adapter: Path | None = None) -> TrainResult:
        raise NotImplementedError("Phase 1 — built BEFORE grpo (8GB-first)")
