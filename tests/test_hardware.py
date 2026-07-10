import pytest

from slm_rl.config.loader import load_tiers
from slm_rl.platform.hardware import HostSpec, detect_host, resolve_tier

TIERS = load_tiers()


def host(os="linux", ram=8.0, vram=None, mps=False) -> HostSpec:
    return HostSpec(os=os, ram_gb=ram, cuda_vram_gb=vram, has_mps=mps)


def test_8gb_mac_gets_universal_floor():
    tier = resolve_tier(TIERS, host(os="darwin", ram=8.0, mps=True))
    assert tier.name == "any-8gb"
    assert tier.train == "reject_sft"


def test_8gb_linux_cpu_gets_universal_floor():
    tier = resolve_tier(TIERS, host(os="linux", ram=8.0))
    assert tier.name == "any-8gb"


def test_16gb_mac_gets_bigger_model():
    tier = resolve_tier(TIERS, host(os="darwin", ram=16.0, mps=True))
    assert tier.name == "mac-16gb"
    assert tier.model == "LiquidAI/LFM2.5-VL-1.6B"


def test_24gb_cuda_tier():
    tier = resolve_tier(TIERS, host(ram=64.0, vram=24.0))
    assert tier.name == "cuda-24gb"


def test_small_cuda_gets_small_model():
    tier = resolve_tier(TIERS, host(ram=32.0, vram=8.0))
    assert tier.name == "cuda-8-16gb"
    assert "Instruct" in tier.model  # ponytail: guards the R2 model-id fix


def test_16gb_linux_without_gpu_falls_to_floor():
    # cuda tiers need VRAM, mac-16gb needs darwin -> universal floor
    tier = resolve_tier(TIERS, host(os="linux", ram=16.0))
    assert tier.name == "any-8gb"


def test_forced_tier_name():
    tier = resolve_tier(TIERS, host(ram=8.0), forced_name="cuda-24gb")
    assert tier.name == "cuda-24gb"
    with pytest.raises(KeyError):
        resolve_tier(TIERS, host(), forced_name="nope")


def test_detect_host_runs_on_this_machine():
    spec = detect_host()
    assert spec.ram_gb > 0
    assert spec.os in ("linux", "darwin", "windows")
    # and the shipped table must resolve for ANY real host
    assert resolve_tier(TIERS, spec) is not None
