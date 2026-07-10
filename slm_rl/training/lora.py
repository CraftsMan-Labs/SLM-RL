"""Shared LoRA helpers: target modules per architecture, and merge-for-export
(`slm-rl export --gen N --merge [--gguf]`, Phase 4)."""

from __future__ import annotations

from pathlib import Path

# LFM2/LFM2.5 is a hybrid conv+attention arch: attention q/k/v + out_proj,
# conv in_proj, GLU w1/w2/w3 (per Unsloth's LFM2.5 guide — note out_proj, not
# o_proj). Substring "lfm2" matches "lfm2.5" too.
LORA_TARGETS = {
    "lfm2": ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def target_modules_for(model_id: str) -> list[str] | str:
    lower = model_id.lower()
    for key, targets in LORA_TARGETS.items():
        if key in lower:
            return targets
    return "all-linear"  # PEFT fallback for unknown architectures


def merge_adapter(base_model_id: str, adapter_path: Path, out_dir: Path, gguf: bool = False) -> Path:
    raise NotImplementedError("Phase 4")
