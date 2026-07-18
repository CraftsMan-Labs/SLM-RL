"""Official HF model presets per hardware tier (plan 026 decision 11).

Official org-owned IDs only (CODING_GUIDELINE §5.4). Rejected: any
Qwen3.6-* small (open weights are 27B / 35B-A3B only — no official small
checkpoint) and community Instruct forks of Qwen3-0.6B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# nvidia/Nemotron-Flash-1B is a base (non-Instruct) model and may need
# trust_remote_code — keep in the list but mark experimental in the UI.
NEMOTRON_FLASH_1B = "nvidia/Nemotron-Flash-1B"

# Orgs allowed in preset lists (tests assert every preset id is under one).
OFFICIAL_ORGS: frozenset[str] = frozenset({"LiquidAI", "Qwen", "google", "nvidia"})


@dataclass(frozen=True)
class ModelPreset:
    model: str
    backend: str | None = None  # suggested when the attendee picks this preset
    experimental: bool = False


# First entry per tier is the default (must match configs/hardware.yaml).
TIER_PRESETS: dict[str, list[ModelPreset]] = {
    "any-8gb": [
        ModelPreset("LiquidAI/LFM2.5-350M", backend="transformers"),
        ModelPreset("Qwen/Qwen3-0.6B", backend="transformers"),
        ModelPreset(NEMOTRON_FLASH_1B, experimental=True),
    ],
    "cuda-8-16gb": [
        ModelPreset("LiquidAI/LFM2.5-1.2B-Instruct", backend="transformers"),
        ModelPreset("Qwen/Qwen2.5-1.5B-Instruct"),
        ModelPreset(NEMOTRON_FLASH_1B, experimental=True),
    ],
    "cuda-24gb": [
        ModelPreset("google/gemma-4-E2B-it", backend="transformers"),
        ModelPreset("LiquidAI/LFM2.5-1.2B-Instruct"),
    ],
    "mac-16gb": [
        ModelPreset("LiquidAI/LFM2.5-1.2B-Instruct", backend="transformers"),
        ModelPreset("LiquidAI/LFM2.5-350M", backend="transformers"),
    ],
}


def preset_label(preset: ModelPreset) -> str:
    """UI option text; Nemotron gets an experimental suffix (plan 026 E)."""
    if preset.experimental:
        return f"{preset.model} (experimental — base; may need trust_remote_code)"
    return preset.model


def presets_for_tier(tier_name: str) -> list[dict[str, Any]]:
    """JSON-ready preset rows for the resolved tier (empty if unknown)."""
    rows = []
    for p in TIER_PRESETS.get(tier_name, []):
        rows.append({
            "model": p.model,
            "backend": p.backend,
            "experimental": p.experimental,
            "label": preset_label(p),
        })
    return rows


def hardware_payload() -> dict[str, Any]:
    """detect_host + resolve_tier + presets for the playground banner/select."""
    from slm_rl.config.loader import load_tiers
    from slm_rl.platform.hardware import detect_host, resolve_tier

    host = detect_host()
    tier = resolve_tier(load_tiers(), host)
    return {
        "tier": tier.name,
        "model": tier.model,
        "backend": tier.backend,
        "train": tier.train,
        "presets": presets_for_tier(tier.name),
        "host": {
            "os": host.os,
            "ram_gb": round(host.ram_gb, 1),
            "cuda_vram_gb": (
                round(host.cuda_vram_gb, 1) if host.cuda_vram_gb is not None else None
            ),
            "has_mps": host.has_mps,
        },
    }


__all__ = [
    "NEMOTRON_FLASH_1B",
    "OFFICIAL_ORGS",
    "ModelPreset",
    "TIER_PRESETS",
    "hardware_payload",
    "preset_label",
    "presets_for_tier",
]
