"""Day-of workshop launcher: detect host/tier → print install + start cmds.

Instructor bootstrap only (CODING_GUIDELINE §5). Attendees stay browser-only.
No in-UI driver wizard — this module prints (and optionally execs) the exact
commands. Prefer ``python -m slm_rl.platform.launch`` over a bash-only script
so the same path works on Linux / macOS / Windows.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from dataclasses import dataclass

from slm_rl.config.loader import load_tiers
from slm_rl.config.schema import TierConfig
from slm_rl.platform.hardware import HostSpec, detect_host, resolve_tier


@dataclass(frozen=True)
class LaunchAdvice:
    """Resolved install + start commands for one host/tier."""

    host: HostSpec
    tier: TierConfig
    install_cmd: str
    start_cmd: str
    docker_cmd: str
    notes: tuple[str, ...]


def uv_extras_for_tier(tier_name: str) -> list[str]:
    """Extras for a workshop-ready bare-metal ``uv sync`` on this tier.

    Always includes ``atari`` (locked ALE slate) and ``dev`` (pytest).
    CUDA and cpu-train are mutually exclusive (pyproject conflicts).
    """
    if tier_name == "mac-16gb":
        return ["atari", "mac", "dev"]
    if tier_name.startswith("cuda"):
        return ["cuda", "atari", "dev"]
    return ["cpu-train", "atari", "dev"]


def docker_compose_cmd(tier_name: str) -> str:
    """Exact compose bring-up for this tier (CPU image vs ``gpu`` profile)."""
    if tier_name.startswith("cuda"):
        return "docker compose --profile gpu up --build playground-gpu"
    return "docker compose up --build"


def advise(
    host: HostSpec | None = None,
    tier: TierConfig | None = None,
    *,
    forced_tier: str | None = None,
) -> LaunchAdvice:
    """Detect (or accept) host/tier and build the install + start command set."""
    host = host or detect_host()
    if tier is None:
        tier = resolve_tier(load_tiers(), host, forced_name=forced_tier)
    extras = uv_extras_for_tier(tier.name)
    install_cmd = "uv sync " + " ".join(f"--extra {e}" for e in extras)
    start_cmd = "uv run slm-rl playground"
    docker_cmd = docker_compose_cmd(tier.name)

    notes: list[str] = [
        f"Resolved tier {tier.name!r} → model={tier.model} backend={tier.backend} "
        f"train={tier.train}",
    ]
    if tier.name == "mac-16gb":
        notes.append(
            "Docker on macOS is CPU-only (Desktop VM); prefer native uv for Metal/MLX."
        )
        notes.append("Bare-metal playground: http://127.0.0.1:8780/")
    elif tier.name.startswith("cuda"):
        notes.append(
            "GPU Docker needs the nvidia container toolkit; "
            "open http://127.0.0.1:8781/ after compose."
        )
        notes.append("Bare-metal playground: http://127.0.0.1:8780/")
    else:
        notes.append(
            "Docker playground bundles cpu-train; open http://127.0.0.1:8780/."
        )
        notes.append("Bare-metal playground: http://127.0.0.1:8780/")
    notes.append(
        "--run execs the start command only (never uv sync). "
        "Use --docker with --run to bring up compose instead."
    )
    return LaunchAdvice(
        host=host,
        tier=tier,
        install_cmd=install_cmd,
        start_cmd=start_cmd,
        docker_cmd=docker_cmd,
        notes=tuple(notes),
    )


def format_advice(advice: LaunchAdvice) -> str:
    """Human-readable block for instructors (stdout)."""
    h = advice.host
    vram = (
        f"{h.cuda_vram_gb:.1f}GB" if h.cuda_vram_gb is not None else "none"
    )
    lines = [
        "=== SLM-RL workshop day-of launcher ===",
        f"host: os={h.os} ram={h.ram_gb:.1f}GB cuda_vram={vram} mps={h.has_mps}",
        f"tier: {advice.tier.name}",
        "",
        "# Install (run once — not executed by --run)",
        advice.install_cmd,
        "",
        "# Or Docker (workshop default on Linux/CI machines)",
        advice.docker_cmd,
        "",
        "# Start playground (bare metal; --run uses this unless --docker)",
        advice.start_cmd,
        "",
        "# Notes",
        *[f"- {n}" for n in advice.notes],
    ]
    return "\n".join(lines)


def _exec_command(cmd: str) -> None:
    """Replace this process with ``cmd`` (never returns on success)."""
    argv = shlex.split(cmd)
    if not argv:
        raise ValueError("empty start command")
    os.execvp(argv[0], argv)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m slm_rl.platform.launch",
        description=(
            "Detect hardware tier and print the exact workshop install + "
            "playground start commands. Instructor bootstrap only."
        ),
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Exec the playground start command (never runs uv sync)",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="With --run, exec docker compose instead of uv run playground",
    )
    parser.add_argument(
        "--tier",
        default=None,
        help="Force a hardware.yaml tier name instead of auto-detect",
    )
    args = parser.parse_args(argv)

    advice = advise(forced_tier=args.tier)
    print(format_advice(advice))

    if not args.run:
        return 0

    cmd = advice.docker_cmd if args.docker else advice.start_cmd
    print(f"\n--run: exec {cmd!r}", flush=True)
    try:
        _exec_command(cmd)
    except FileNotFoundError as exc:
        print(f"error: cannot exec {cmd!r}: {exc}", file=sys.stderr)
        return 1
    return 0  # unreachable after successful execvp


if __name__ == "__main__":
    raise SystemExit(main())
