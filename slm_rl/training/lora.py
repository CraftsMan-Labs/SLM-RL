"""Shared LoRA helpers: adapter save/load conventions, merge-for-export
(`slm-rl export --gen N --merge [--gguf]`)."""

from __future__ import annotations

from pathlib import Path


def merge_adapter(base_model_id: str, adapter_path: Path, out_dir: Path, gguf: bool = False) -> Path:
    raise NotImplementedError("Phase 4")
