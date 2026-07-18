"""Streaming JSONL writer — records are flushed as they happen, never
accumulated in RAM (8GB budget rule)."""

from __future__ import annotations

from pathlib import Path

from slm_rl.datagen.schema import RolloutRecord


class RolloutWriter:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate: a restarted generation must not append onto stale episodes
        # (that inflated GRPO prompt counts into multi-hour Docker-CPU trains).
        self._file = open(self.path, "w", encoding="utf-8")

    def write(self, record: RolloutRecord) -> None:
        self._file.write(record.to_json() + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "RolloutWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
