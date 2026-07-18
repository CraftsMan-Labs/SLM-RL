"""Hardware detection -> tier resolution against configs/hardware.yaml.

Never hardcodes models: detection only resolves against the config-driven
tier table. First matching tier wins, so the table is ordered from most to
least capable, ending with the universal `any-8gb` floor.
"""

from __future__ import annotations

import platform as _platform
import sys
from dataclasses import dataclass

import psutil

from slm_rl.config.schema import TierConfig


@dataclass(frozen=True)
class HostSpec:
    os: str  # "darwin" | "linux" | "windows"
    ram_gb: float
    cuda_vram_gb: float | None  # None = no CUDA
    has_mps: bool


def detect_host() -> HostSpec:
    os_name = {"darwin": "darwin", "linux": "linux", "win32": "windows"}.get(
        sys.platform, sys.platform
    )
    ram_gb = psutil.virtual_memory().total / 1024**3

    cuda_vram_gb: float | None = None
    has_mps = False
    try:
        import torch

        if torch.cuda.is_available():
            cuda_vram_gb = (
                torch.cuda.get_device_properties(0).total_memory / 1024**3
            )
        has_mps = bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        )
    except ImportError:
        # torch is an optional extra; without it we may still have a GPU —
        # try nvidia-smi as a dependency-free fallback.
        cuda_vram_gb = _vram_from_nvidia_smi()
        has_mps = os_name == "darwin" and _platform.machine() == "arm64"

    return HostSpec(os=os_name, ram_gb=ram_gb, cuda_vram_gb=cuda_vram_gb, has_mps=has_mps)


def _vram_from_nvidia_smi() -> float | None:
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
        mib = float(out.strip().splitlines()[0])
        return mib / 1024
    except Exception:
        return None


def tier_matches(tier: TierConfig, host: HostSpec) -> bool:
    if tier.os is not None and tier.os != host.os:
        return False
    if tier.min_ram_gb is not None and host.ram_gb < tier.min_ram_gb:
        return False
    if tier.max_ram_gb is not None and host.ram_gb > tier.max_ram_gb:
        return False
    if tier.min_cuda_vram_gb is not None:
        if host.cuda_vram_gb is None or host.cuda_vram_gb < tier.min_cuda_vram_gb:
            return False
    if tier.requires_mps is not None and tier.requires_mps != host.has_mps:
        return False
    return True


def resolve_tier(
    tiers: list[TierConfig],
    host: HostSpec | None = None,
    forced_name: str | None = None,
) -> TierConfig:
    host = host or detect_host()
    if forced_name:
        for tier in tiers:
            if tier.name == forced_name:
                return tier
        raise KeyError(f"No tier named {forced_name!r} in hardware.yaml")
    for tier in tiers:
        if tier_matches(tier, host):
            return tier
    raise RuntimeError(
        f"No tier matches this host ({host}). hardware.yaml must end with a "
        "universal fallback tier with no match conditions."
    )
