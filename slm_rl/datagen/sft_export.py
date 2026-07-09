"""Trajectory selection + export to SFT chat datasets for the reject_sft
strategy: keep winning / top-quantile, monitor-clean trajectories with a
diversity quota (see docs/DECISIONS.md D10)."""

from __future__ import annotations

from pathlib import Path

from slm_rl.config.schema import TrainConfig


def export_sft_dataset(dataset_path: Path, out_path: Path, cfg: TrainConfig) -> int:
    """Returns the number of (prompt -> action) pairs written."""
    raise NotImplementedError("Phase 1")
