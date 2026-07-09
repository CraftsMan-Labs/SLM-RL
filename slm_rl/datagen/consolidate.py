"""JSONL -> parquet consolidation (chunked; 8GB budget rule). Requires the
[dashboard] extra (pyarrow/pandas)."""

from __future__ import annotations

from pathlib import Path


def consolidate(rollout_dir: Path, out_path: Path, chunk_rows: int = 50_000) -> None:
    raise NotImplementedError("Phase 1")
