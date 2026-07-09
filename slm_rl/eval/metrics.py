"""Metrics aggregation: per-generation metrics.json (train + eval +
doom-loop stats) consumed by the dashboard and the EvalGate."""

from __future__ import annotations

from pathlib import Path


def write_metrics(path: Path, metrics: dict) -> None:
    raise NotImplementedError("Phase 1")
