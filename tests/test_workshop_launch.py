"""Tests for ``python -m slm_rl.platform.launch`` (plan 026 Phase H)."""

from __future__ import annotations

import pytest

from slm_rl.config.schema import TierConfig
from slm_rl.platform.hardware import HostSpec
from slm_rl.platform.launch import (
    advise,
    docker_compose_cmd,
    format_advice,
    main,
    uv_extras_for_tier,
)


def _tier(name: str, model: str = "LiquidAI/LFM2.5-350M") -> TierConfig:
    return TierConfig(
        name=name,
        model=model,
        backend="transformers",
        train="grpo",
    )


def test_uv_extras_any_8gb_includes_cpu_train_and_atari():
    extras = uv_extras_for_tier("any-8gb")
    assert extras == ["cpu-train", "atari", "dev"]


def test_uv_extras_mac_and_cuda():
    assert uv_extras_for_tier("mac-16gb") == ["atari", "mac", "dev"]
    assert uv_extras_for_tier("cuda-8-16gb") == ["cuda", "atari", "dev"]
    assert uv_extras_for_tier("cuda-24gb") == ["cuda", "atari", "dev"]


def test_docker_compose_gpu_profile_only_on_cuda():
    assert docker_compose_cmd("any-8gb") == "docker compose up --build"
    assert docker_compose_cmd("mac-16gb") == "docker compose up --build"
    assert (
        docker_compose_cmd("cuda-24gb")
        == "docker compose --profile gpu up --build playground-gpu"
    )


def test_advise_prints_exact_install_and_start_for_floor():
    host = HostSpec(os="linux", ram_gb=8.0, cuda_vram_gb=None, has_mps=False)
    advice = advise(host=host, tier=_tier("any-8gb"))
    assert advice.install_cmd == (
        "uv sync --extra cpu-train --extra atari --extra dev"
    )
    assert advice.start_cmd == "uv run slm-rl playground"
    assert advice.docker_cmd == "docker compose up --build"
    text = format_advice(advice)
    assert advice.install_cmd in text
    assert advice.start_cmd in text
    assert advice.docker_cmd in text


def test_advise_cuda_points_at_gpu_compose():
    host = HostSpec(os="linux", ram_gb=32.0, cuda_vram_gb=24.0, has_mps=False)
    advice = advise(
        host=host,
        tier=_tier("cuda-24gb", model="google/gemma-4-E2B-it"),
    )
    assert "--extra cuda" in advice.install_cmd
    assert "--profile gpu" in advice.docker_cmd
    assert "cpu-train" not in advice.install_cmd  # conflicts with cuda


def test_main_without_run_does_not_exec(capsys, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        "slm_rl.platform.launch._exec_command",
        lambda cmd: calls.append(cmd),
    )
    monkeypatch.setattr(
        "slm_rl.platform.launch.advise",
        lambda **kwargs: advise(
            host=HostSpec(os="linux", ram_gb=8.0, cuda_vram_gb=None, has_mps=False),
            tier=_tier("any-8gb"),
        ),
    )
    assert main([]) == 0
    assert calls == []
    out = capsys.readouterr().out
    assert "uv sync --extra cpu-train" in out
    assert "uv run slm-rl playground" in out


def test_main_run_execs_start_not_sync(monkeypatch):
    calls: list[str] = []

    def fake_exec(cmd: str) -> None:
        calls.append(cmd)
        raise SystemExit(0)  # simulate successful exec never returning

    monkeypatch.setattr("slm_rl.platform.launch._exec_command", fake_exec)
    monkeypatch.setattr(
        "slm_rl.platform.launch.advise",
        lambda **kwargs: advise(
            host=HostSpec(os="linux", ram_gb=8.0, cuda_vram_gb=None, has_mps=False),
            tier=_tier("any-8gb"),
        ),
    )
    with pytest.raises(SystemExit) as exc:
        main(["--run"])
    assert exc.value.code == 0
    assert calls == ["uv run slm-rl playground"]
    assert not any("uv sync" in c for c in calls)


def test_main_run_docker_execs_compose(monkeypatch):
    calls: list[str] = []

    def fake_exec(cmd: str) -> None:
        calls.append(cmd)
        raise SystemExit(0)

    monkeypatch.setattr("slm_rl.platform.launch._exec_command", fake_exec)
    monkeypatch.setattr(
        "slm_rl.platform.launch.advise",
        lambda **kwargs: advise(
            host=HostSpec(os="linux", ram_gb=64.0, cuda_vram_gb=24.0, has_mps=False),
            tier=_tier("cuda-24gb", model="google/gemma-4-E2B-it"),
        ),
    )
    with pytest.raises(SystemExit):
        main(["--run", "--docker"])
    assert calls == [
        "docker compose --profile gpu up --build playground-gpu",
    ]
