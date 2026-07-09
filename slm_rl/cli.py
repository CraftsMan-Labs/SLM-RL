"""SLM-RL command line. `info` works today; the pipeline commands land with
their phases (see docs/ROADMAP.md)."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="slm-rl",
    help="A self-improving game gymnasium for small language models.",
    no_args_is_help=True,
)

_SKELETON_MSG = "Not implemented yet — lands in {phase}. See docs/ROADMAP.md."


def _todo(phase: str) -> None:
    typer.secho(_SKELETON_MSG.format(phase=phase), fg=typer.colors.YELLOW)
    raise typer.Exit(1)


@app.command()
def info(tier: str = typer.Option(None, help="Force a tier by name instead of auto-detecting")) -> None:
    """Show detected hardware, the resolved tier, and available games."""
    from slm_rl.config.loader import load_tiers
    from slm_rl.games.registry import available_games
    from slm_rl.platform.hardware import detect_host, resolve_tier

    host = detect_host()
    typer.echo(f"host: os={host.os} ram={host.ram_gb:.1f}GB "
               f"cuda_vram={host.cuda_vram_gb and f'{host.cuda_vram_gb:.1f}GB'} mps={host.has_mps}")
    resolved = resolve_tier(load_tiers(), host, forced_name=tier)
    typer.echo(f"tier: {resolved.name} -> model={resolved.model} "
               f"backend={resolved.backend} train={resolved.train}")
    typer.echo(f"games: {', '.join(available_games())}")


@app.command()
def play(game: str = typer.Argument("mastermind")) -> None:
    """Play a game interactively against the current champion."""
    _todo("Phase 1")


@app.command()
def rollout(
    game: str = typer.Option("mastermind"),
    agent: str = typer.Option("llm", help="llm | random"),
    episodes: int = typer.Option(10),
) -> None:
    """Generate rollouts (dataset product) without training."""
    _todo("Phase 1")


@app.command()
def train(game: str = typer.Option("mastermind")) -> None:
    """Run one training step on collected rollouts."""
    _todo("Phase 1")


@app.command()
def evolve(
    game: str = typer.Option("mastermind"),
    generations: int = typer.Option(5),
) -> None:
    """Run the full self-improvement loop for N generations."""
    _todo("Phase 1")


@app.command("eval")
def eval_cmd(game: str = typer.Option("mastermind")) -> None:
    """Run the frozen eval suite for a game."""
    _todo("Phase 1")


@app.command()
def elo(game: str = typer.Option("connect4")) -> None:
    """Show the ELO league for a competitive game."""
    _todo("Phase 2")


@app.command()
def export(
    gen: int = typer.Option(..., help="Generation to export"),
    merge: bool = typer.Option(False),
    gguf: bool = typer.Option(False),
) -> None:
    """Export a generation's model (merged / GGUF for Mac play)."""
    _todo("Phase 4")


@app.command()
def dashboard() -> None:
    """Launch the metrics dashboard."""
    _todo("Phase 4")


@app.command("new-game")
def new_game(name: str = typer.Argument(...)) -> None:
    """Scaffold a new game plugin."""
    _todo("Phase 4")


if __name__ == "__main__":
    app()
