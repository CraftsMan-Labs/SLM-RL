"""TrainingStrategy: the swappable TRAIN box of the generation loop (D10).

Implementations:
- `grpo` (training/grpo.py): TRL GRPOTrainer + PEFT LoRA — CUDA tiers.
- `reject_sft` (training/reject_sft.py): rejection-sampling SFT — the
  universal 8GB path (mlx-lm on Mac, transformers+PEFT elsewhere).

Both consume the same parquet dataset and produce a LoRA adapter directory;
the orchestrator's EvalGate treats them identically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slm_rl.config.schema import GameConfig, TrainConfig


@dataclass
class TrainResult:
    adapter_path: Path
    metrics: dict[str, Any] = field(default_factory=dict)  # loss curve, mean entropy, ...


class TrainingStrategy(ABC):
    name: str

    def __init__(self, cfg: TrainConfig, model_id: str, game_cfg: GameConfig | None = None):
        self.cfg = cfg
        self.model_id = model_id
        self.game_cfg = game_cfg  # needed by grpo (env-grounded rewards); reject_sft ignores it

    @abstractmethod
    def train(
        self,
        dataset_path: Path,
        out_dir: Path,
        init_adapter: Path | None = None,  # previous champion's adapter
    ) -> TrainResult: ...


def create_strategy(
    name: str, cfg: TrainConfig, model_id: str, game_cfg: GameConfig | None = None
) -> TrainingStrategy:
    """Lazy factory — heavy imports stay inside the strategy modules."""
    if name == "grpo":
        from slm_rl.training.grpo import GRPOStrategy

        return GRPOStrategy(cfg, model_id, game_cfg)
    if name == "reject_sft":
        from slm_rl.training.reject_sft import RejectSFTStrategy

        return RejectSFTStrategy(cfg, model_id, game_cfg)
    raise ValueError(f"Unknown training strategy {name!r}")
