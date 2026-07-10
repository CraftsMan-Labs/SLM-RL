"""JSONL -> parquet consolidation, chunked (8GB budget rule).

Nested fields (prompt_messages, monitor_flags) are stored as JSON strings so
the parquet schema stays stable across games and schema versions. Requires
the [dashboard] extra (pyarrow).
"""

from __future__ import annotations

import json
from pathlib import Path


def consolidate(rollout_dir: Path | str, out_path: Path | str, chunk_rows: int = 50_000) -> int:
    """Consolidates all *.jsonl under rollout_dir; returns rows written."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rollout_dir, out_path = Path(rollout_dir), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Explicit schema so chunk boundaries can't change inferred types
    # (e.g. an all-null `outcome` chunk inferring null vs string). All nested
    # fields are JSON-stringified below, so everything is scalar except the
    # legal_actions list.
    schema = pa.schema([
        ("run_id", pa.string()), ("generation", pa.int64()), ("game", pa.string()),
        ("episode_id", pa.string()), ("step_idx", pa.int64()), ("seed", pa.int64()),
        ("model_id", pa.string()), ("adapter_ref", pa.string()), ("opponent_id", pa.string()),
        ("prompt_messages", pa.string()), ("completion", pa.string()),
        ("parsed_action", pa.string()), ("legal_actions", pa.list_(pa.string())),
        ("parse_status", pa.string()), ("reward", pa.float64()),
        ("shaped_reward", pa.float64()), ("cum_reward", pa.float64()),
        ("terminated", pa.bool_()), ("truncated", pa.bool_()), ("outcome", pa.string()),
        ("state_hash", pa.string()), ("monitor_flags", pa.string()),
        ("timestamp", pa.string()), ("schema_version", pa.int64()),
    ])
    writer = pq.ParquetWriter(out_path, schema)
    batch: list[dict] = []
    total = 0

    def flush() -> None:
        nonlocal batch, total
        if not batch:
            return
        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        total += len(batch)
        batch = []

    for jsonl in sorted(rollout_dir.glob("*.jsonl")):
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["prompt_messages"] = json.dumps(row["prompt_messages"], ensure_ascii=False)
                row["monitor_flags"] = json.dumps(row["monitor_flags"], ensure_ascii=False)
                batch.append(row)
                if len(batch) >= chunk_rows:
                    flush()
    flush()
    writer.close()
    return total
