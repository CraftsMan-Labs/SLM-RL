"""Locate on-disk DQN teacher checkpoints (bake packs + train-dqn output).

Playground materialization and `make_teacher` share this so "teacher=dqn"
never silently points every game at Space Invaders.
"""

from __future__ import annotations

from pathlib import Path


def expected_dqn_checkpoint(game: str, home: Path | str | None = None) -> Path:
    """Canonical `train-dqn` output path for `game` (may not exist yet)."""
    root = Path(home) if home is not None else Path("runs")
    return root / "teachers" / f"dqn-{game}.pt"


def find_dqn_checkpoint(game: str, home: Path | str | None = None) -> Path | None:
    """Return an existing checkpoint for `game`, or None.

    Search order under each root:
      1. `<root>/packs/<game>/dqn.pt` (workshop bake)
      2. `<root>/teachers/dqn-<game>.pt` (`slm-rl train-dqn`)

    If `home` is given, only that root is searched. If omitted, `./runs`
    (relative to cwd) is used — the Docker / bare-metal convention.
    """
    roots = [Path(home)] if home is not None else [Path("runs")]

    for root in roots:
        for candidate in (
            root / "packs" / game / "dqn.pt",
            root / "teachers" / f"dqn-{game}.pt",
        ):
            if candidate.is_file():
                return candidate.resolve()
    return None


def is_legacy_space_invaders_default(path: str | Path, game: str) -> bool:
    """True when an older playground wrote the SI checkpoint for every game."""
    name = Path(path).name
    return name == "dqn-space-invaders.pt" and game != "space-invaders"


def missing_dqn_hint(game: str, home: Path | str | None = None) -> str:
    """Operator-facing message when teacher=dqn but no file is on disk."""
    expected = expected_dqn_checkpoint(game, home)
    return (
        f"No DQN checkpoint for {game!r}. Bake a workshop pack for this game "
        f"(Projects → Bake workshop packs), or train one:\n"
        f"  slm-rl train-dqn --game {game} --out {expected}"
    )
