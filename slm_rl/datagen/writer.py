"""Streaming JSONL writer — records are flushed as they happen, never
accumulated in RAM (8GB budget rule)."""

from __future__ import annotations

from pathlib import Path

from slm_rl.datagen.schema import RolloutRecord


class RolloutWriter:
    def __init__(self, path: Path):
        raise NotImplementedError("Phase 1")

    def write(self, record: RolloutRecord) -> None:
        raise NotImplementedError("Phase 1")

    def close(self) -> None:
        raise NotImplementedError("Phase 1")
