"""Optional antidoom hygiene stage (Liquid4All/antidoom, FTPO): run between
generations when the EvalGate's auto-remediation triggers, to strip
degenerate repetition loops from the model. Off by default; invoked as a
subprocess around the external `antidoom` tool (generate/detect/train/merge)."""

from __future__ import annotations

from pathlib import Path


def run_antidoom_stage(model_path: Path, out_dir: Path) -> Path:
    raise NotImplementedError("Phase 3 (optional, gated by auto-remediation)")
