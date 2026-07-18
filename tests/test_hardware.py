import inspect

import pytest

from slm_rl.config.loader import load_tiers
from slm_rl.inference.base import create_backend
from slm_rl.platform.hardware import HostSpec, detect_host, resolve_tier

TIERS = load_tiers()


def host(os="linux", ram=8.0, vram=None, mps=False) -> HostSpec:
    return HostSpec(os=os, ram_gb=ram, cuda_vram_gb=vram, has_mps=mps)


def test_8gb_mac_gets_universal_floor():
    tier = resolve_tier(TIERS, host(os="darwin", ram=8.0, mps=True))
    assert tier.name == "any-8gb"
    assert tier.backend == "transformers"
    assert tier.train == "grpo"


def test_8gb_linux_cpu_gets_universal_floor():
    tier = resolve_tier(TIERS, host(os="linux", ram=8.0))
    assert tier.name == "any-8gb"


def test_16gb_mac_gets_bigger_model():
    tier = resolve_tier(TIERS, host(os="darwin", ram=16.0, mps=True))
    assert tier.name == "mac-16gb"
    assert tier.model == "LiquidAI/LFM2.5-1.2B-Instruct"


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


def _resolves_to_real_class(name: str):
    """Import (never instantiate — mlx pulls real deps and would try real
    model/network resolution) the backend class create_backend would return,
    and fail loudly if its methods are stub bodies. Guards against plan
    024's regression: every backend string in hardware.yaml silently raising
    NotImplementedError on first use."""
    import importlib

    module_for = {
        "transformers": "slm_rl.inference.transformers_be",
        "transformers-4bit": "slm_rl.inference.transformers_be",
        "mlx": "slm_rl.inference.mlx_be",
    }
    class_for = {
        "transformers": "TransformersBackend",
        "transformers-4bit": "TransformersBackend",
        "mlx": "MLXBackend",
    }
    mod = importlib.import_module(module_for[name])
    return getattr(mod, class_for[name])


def _is_stub(cls) -> bool:
    """A stub's __init__/generate body is (module-doc, raise NotImplementedError)
    and nothing else -- the exact plan-020 post-mortem pattern. A real
    implementation may still mention NotImplementedError (e.g. a documented,
    narrow gap like load_adapter) without being a stub overall: we only
    check __init__ and generate, the two methods every backend must do real
    work in."""
    for meth_name in ("__init__", "generate"):
        src = inspect.getsource(getattr(cls, meth_name))
        body_lines = [
            line.strip() for line in src.splitlines()[1:]  # skip the def line
            if line.strip() and not line.strip().startswith(("#", '"""'))
        ]
        if len(body_lines) <= 1 and any("NotImplementedError" in line for line in body_lines):
            return True
    return False


@pytest.mark.parametrize("tier_name", [t.name for t in TIERS])
def test_tier_backend_resolves_to_a_real_class_not_a_stub(tier_name):
    """Walks every tier in hardware.yaml so a future stubbed-out backend can
    never silently ship again (plan 024)."""
    tier = next(t for t in TIERS if t.name == tier_name)
    # transformers backends need torch installed (cuda/cpu-train extras).
    if tier.backend in ("transformers", "transformers-4bit"):
        pytest.importorskip("torch")
    if tier.backend == "mlx":
        pytest.importorskip("mlx_lm")
    try:
        cls = _resolves_to_real_class(tier.backend)
    except ModuleNotFoundError as exc:
        pytest.skip(f"optional backend dep missing for {tier.backend!r}: {exc}")
    assert not _is_stub(cls), (
        f"{tier.backend!r} (tier {tier_name!r}) is still a Phase-1 stub — "
        f"{cls.__module__}.{cls.__name__} raises NotImplementedError unconditionally"
    )
