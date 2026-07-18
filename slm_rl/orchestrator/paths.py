"""On-disk layout for runs (see docs/ARCHITECTURE.md):

runs/<run_id>/
  run_config.yaml            frozen resolved config
  registry.json              champion pointer + history + ELO
  generations/gen_NNN/
    adapter/                 PEFT LoRA adapter (unmerged)
    rollouts/*.jsonl         raw per-episode step records
    dataset/train.parquet    consolidated training view
    eval/results.json
    metrics.json
    MANIFEST.json            base model id, parent gen, config hash, git sha
"""

from __future__ import annotations

from pathlib import Path


class RunPaths:
    def __init__(self, home: Path | str, run_id: str):
        self.root = Path(home) / run_id

    @property
    def registry(self) -> Path:
        return self.root / "registry.json"

    def generation(self, gen: int) -> Path:
        return self.root / "generations" / f"gen_{gen:03d}"

    def adapter(self, gen: int) -> Path:
        return self.generation(gen) / "adapter"

    def rollouts(self, gen: int) -> Path:
        return self.generation(gen) / "rollouts"

    def dataset(self, gen: int) -> Path:
        return self.generation(gen) / "dataset" / "train.parquet"

    def metrics(self, gen: int) -> Path:
        return self.generation(gen) / "metrics.json"

    def manifest(self, gen: int) -> Path:
        return self.generation(gen) / "MANIFEST.json"
