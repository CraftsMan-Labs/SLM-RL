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


def _resolve_model_backend(model, backend, tier):
    """Resolve (model_id, backend_name, quantization) from a tier + overrides."""
    from slm_rl.config.loader import load_tiers
    from slm_rl.platform.hardware import resolve_tier

    t = resolve_tier(load_tiers(), forced_name=tier)
    return model or t.model, backend or t.backend, t.quantization


def _build_llm_agent(game_cls, game_cfg, model, backend, tier, adapter, temperature):
    from slm_rl.agents.llm_agent import LLMAgent
    from slm_rl.inference.base import GenParams, create_backend

    model_id, backend_name, quant = _resolve_model_backend(model, backend, tier)
    be = create_backend(backend_name, model_id, quant)
    if adapter:
        be.load_adapter(adapter)
    params = GenParams(max_tokens=256, temperature=temperature)
    agent = LLMAgent(be, game_cls(game_cfg).system_prompt(), gen_params=params)
    return agent, be, model_id


@app.command()
def rollout(
    game: str = typer.Option("mastermind"),
    agent: str = typer.Option("random", help="llm | random"),
    episodes: int = typer.Option(10),
    seed: int = typer.Option(0, help="First episode seed (consecutive after)"),
    run_id: str = typer.Option("adhoc"),
    generation: int = typer.Option(0),
    model: str = typer.Option(None),
    backend: str = typer.Option(None),
    tier: str = typer.Option(None),
    adapter: str = typer.Option(None, help="LoRA adapter path (llm agent)"),
) -> None:
    """Generate rollouts (dataset product) without training."""
    from slm_rl.agents.bots import RandomAgent
    from slm_rl.config.loader import load_game_config, load_run_config
    from slm_rl.datagen.writer import RolloutWriter
    from slm_rl.games.registry import get_game
    from slm_rl.orchestrator.paths import RunPaths
    from slm_rl.rollout.runner import EpisodeRunner

    run_cfg = load_run_config(game=game)
    game_cfg = load_game_config(game)
    game_cls = get_game(game)
    paths = RunPaths(run_cfg.home, run_id)
    out = paths.rollouts(generation) / f"{game}-seed{seed}.jsonl"

    backend_obj = None
    model_id = "random"
    if agent == "llm":
        llm, backend_obj, model_id = _build_llm_agent(
            game_cls, game_cfg, model, backend, tier, adapter, temperature=0.8
        )
        make_agent = lambda i: llm  # noqa: E731 (stateless per turn -> reuse)
    else:
        make_agent = lambda i: RandomAgent(seed=seed + i)  # noqa: E731

    wins = 0
    try:
        with RolloutWriter(out) as writer:
            for i in range(episodes):
                runner = EpisodeRunner(
                    game_cls(game_cfg), make_agent(i), game_cfg, writer=writer,
                    run_id=run_id, generation=generation, model_id=model_id,
                    adapter_ref=adapter,
                )
                summary = runner.run_episode(seed + i, episode_id=f"{game}-{seed + i}")
                wins += summary["outcome"] == "win"
                typer.echo(
                    f"episode {i + 1}/{episodes} seed={seed + i} "
                    f"outcome={summary['outcome']} steps={summary['steps']} "
                    f"reward={summary['cum_reward']:.2f}"
                )
    finally:
        if backend_obj:
            backend_obj.close()
    typer.echo(f"win rate: {wins}/{episodes} -> {out}")


@app.command()
def train(
    game: str = typer.Option("mastermind"),
    gen: int = typer.Option(0, help="Generation whose rollouts to train on"),
    run_id: str = typer.Option("adhoc"),
    model: str = typer.Option(None),
    tier: str = typer.Option(None),
    train_strategy: str = typer.Option(None, help="grpo | reject_sft (default: tier's)"),
) -> None:
    """Run the training strategy on an existing generation's rollouts."""
    from slm_rl.config.loader import load_game_config, load_run_config
    from slm_rl.orchestrator.paths import RunPaths
    from slm_rl.platform.hardware import resolve_tier
    from slm_rl.config.loader import load_tiers
    from slm_rl.datagen.consolidate import consolidate
    from slm_rl.training.base import create_strategy

    run_cfg = load_run_config(game=game)
    t = resolve_tier(load_tiers(), forced_name=tier)
    model_id = model or t.model
    paths = RunPaths(run_cfg.home, run_id)

    dataset = paths.dataset(gen)
    if not dataset.exists():
        consolidate(paths.rollouts(gen), dataset)
    strategy = create_strategy(
        train_strategy or run_cfg.train_strategy or t.train,
        run_cfg.train, model_id, load_game_config(game),
    )
    result = strategy.train(dataset, paths.generation(gen))
    typer.echo(f"trained -> {result.adapter_path}")
    typer.echo(f"metrics: {result.metrics}")


@app.command()
def evolve(
    game: str = typer.Option("mastermind"),
    generations: int = typer.Option(5),
    run_id: str = typer.Option("default"),
    model: str = typer.Option(None),
    backend: str = typer.Option(None),
    tier: str = typer.Option(None),
    episodes: int = typer.Option(None, help="Override episodes per generation"),
    train_strategy: str = typer.Option(None, help="grpo | reject_sft (default: tier's)"),
) -> None:
    """Run the full self-improvement loop for N generations."""
    from slm_rl.config.loader import load_run_config
    from slm_rl.orchestrator.generation import GenerationRunner

    overrides = {
        "run_id": run_id, "model": model, "backend": backend, "tier": tier,
        "train_strategy": train_strategy,
    }
    if episodes:
        overrides["train"] = {"episodes_per_generation": episodes}
    cfg = load_run_config(game=game, overrides=overrides)
    runner = GenerationRunner(cfg)
    runner.ensure_baseline()
    start = runner.registry.next_generation
    for g in range(start, start + generations):
        m = runner.run_generation(g)
        typer.echo(
            f"gen {g}: primary={m['eval']['primary']:.3f} "
            f"promoted={m['gate']['promoted']} ({m['gate']['reason']})"
        )


@app.command("eval")
def eval_cmd(
    game: str = typer.Option("mastermind"),
    agent: str = typer.Option("random", help="llm | random"),
    limit: int = typer.Option(None, help="Cap suite episodes (smoke tests)"),
    model: str = typer.Option(None),
    backend: str = typer.Option(None),
    tier: str = typer.Option(None),
    adapter: str = typer.Option(None, help="LoRA adapter path (llm agent)"),
) -> None:
    """Run the frozen eval suite for a game."""
    from slm_rl.agents.bots import RandomAgent
    from slm_rl.config.loader import load_game_config
    from slm_rl.eval.suites import run_suite
    from slm_rl.games.registry import get_game

    game_cls = get_game(game)
    game_cfg = load_game_config(game)

    backend_obj = None
    if agent == "llm":
        llm, backend_obj, _ = _build_llm_agent(
            game_cls, game_cfg, model, backend, tier, adapter, temperature=0.2
        )
        make_agent = lambda: llm  # noqa: E731
    else:
        make_agent = RandomAgent

    try:
        metrics = run_suite(game_cls.eval_suite(), make_agent, game_cls, game_cfg, limit=limit)
    finally:
        if backend_obj:
            backend_obj.close()
    for key, value in metrics.items():
        typer.echo(f"{key}: {value}")


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
