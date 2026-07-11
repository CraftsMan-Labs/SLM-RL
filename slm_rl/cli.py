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
    agent: str = typer.Option("random", help="llm | random | solver (teacher)"),
    episodes: int = typer.Option(10),
    seed: int = typer.Option(0, help="First episode seed (consecutive after)"),
    run_id: str = typer.Option("adhoc"),
    generation: int = typer.Option(0),
    model: str = typer.Option(None),
    backend: str = typer.Option(None),
    tier: str = typer.Option(None),
    adapter: str = typer.Option(None, help="LoRA adapter path (llm agent)"),
    pruner: bool = typer.Option(False, "--pruner/--no-pruner", help="Teacher menu pruning"),
    dqn_checkpoint: str = typer.Option(None, help="DQN teacher checkpoint (agent=solver only)"),
    config_dir: str = typer.Option(None, help="Alternate configs/ root (playground experiments)"),
) -> None:
    """Generate rollouts (dataset product) without training."""
    from pathlib import Path

    from slm_rl.agents.bots import RandomAgent
    from slm_rl.config.loader import load_game_config, load_run_config
    from slm_rl.datagen.writer import RolloutWriter
    from slm_rl.games.registry import get_game
    from slm_rl.orchestrator.paths import RunPaths
    from slm_rl.rollout.runner import EpisodeRunner

    cfg_dir = Path(config_dir) if config_dir else None
    run_cfg = load_run_config(game=game, config_dir=cfg_dir)
    game_cfg = load_game_config(game, config_dir=cfg_dir)
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
    elif agent == "solver":
        from slm_rl.teachers import make_teacher

        solver, model_id = make_teacher(game_cfg, seed=seed, dqn_checkpoint=dqn_checkpoint)
        make_agent = lambda i: solver  # noqa: E731
    else:
        make_agent = lambda i: RandomAgent(seed=seed + i)  # noqa: E731

    pruner_obj = None
    if pruner:
        from slm_rl.teachers import make_pruner

        pruner_obj = make_pruner(game_cfg, top_k=run_cfg.teacher.pruner_top_k)

    wins = 0
    try:
        with RolloutWriter(out) as writer:
            for i in range(episodes):
                runner = EpisodeRunner(
                    game_cls(game_cfg), make_agent(i), game_cfg, writer=writer,
                    run_id=run_id, generation=generation, model_id=model_id,
                    adapter_ref=adapter, pruner=pruner_obj,
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


@app.command("train-dqn")
def train_dqn_cmd(
    game: str = typer.Option(..., help="Game to train the DQN teacher on (e.g. space-invaders)"),
    decisions: int = typer.Option(500_000, help="Training decisions (engine-speed steps)"),
    out: str = typer.Option(..., help="Checkpoint output path"),
    device: str = typer.Option("cpu", help="torch device (cpu | cuda)"),
    seed: int = typer.Option(0),
) -> None:
    """Train a CleanRL-pattern DQN teacher over Game.vector_obs() (plan 012)."""
    from slm_rl.config.loader import load_game_config
    from slm_rl.teachers.dqn import train_dqn

    game_cfg = load_game_config(game)
    summary = train_dqn(game_cfg, decisions=decisions, out_path=out, device=device, seed=seed)
    typer.echo(f"summary: {summary}")


@app.command()
def evolve(
    game: str = typer.Option("mastermind"),
    generations: int = typer.Option(5),
    run_id: str = typer.Option("default"),
    model: str = typer.Option(None),
    backend: str = typer.Option(None),
    tier: str = typer.Option(None),
    episodes: int = typer.Option(None, help="Override episodes per generation"),
    rollout_batch_size: int = typer.Option(
        None,
        help="Override train.rollout_batch_size (K episodes per generate call; transformers/vLLM only)",
    ),
    selection_quantile: float = typer.Option(
        None,
        help="Override train.selection_quantile (keep top fraction of episodes by return for SFT)",
    ),
    train_strategy: str = typer.Option(None, help="grpo | reject_sft (default: tier's)"),
    hf_repo: str = typer.Option(None, help="Push each generation's datasets to this HF dataset repo"),
    pruner: bool = typer.Option(False, "--pruner/--no-pruner", help="Teacher menu pruning during rollout (HYBRID_RL.md)"),
    warm_start: bool = typer.Option(False, "--warm-start", help="Gen 1 = teacher rollout distilled via reject_sft"),
    config_dir: str = typer.Option(None, help="Alternate configs/ root (playground experiments)"),
) -> None:
    """Run the full self-improvement loop for N generations."""
    from pathlib import Path

    from slm_rl.config.loader import load_run_config
    from slm_rl.orchestrator.generation import GenerationRunner

    cfg_dir = Path(config_dir) if config_dir else None
    overrides = {
        "run_id": run_id, "model": model, "backend": backend, "tier": tier,
        "train_strategy": train_strategy, "hf_dataset_repo": hf_repo,
        "teacher": {"pruner": pruner} if pruner else None,
    }
    train_overrides = {}
    if episodes:
        train_overrides["episodes_per_generation"] = episodes
    if rollout_batch_size:
        train_overrides["rollout_batch_size"] = rollout_batch_size
    if selection_quantile:
        train_overrides["selection_quantile"] = selection_quantile
    if train_overrides:
        overrides["train"] = train_overrides
    cfg = load_run_config(game=game, overrides=overrides, config_dir=cfg_dir)
    runner = GenerationRunner(cfg, config_dir=cfg_dir)
    runner.ensure_baseline()
    start = runner.registry.next_generation
    end = start + generations
    if warm_start:
        if start == 1:
            m = runner.run_generation(1, teacher=True)
            typer.echo(
                f"gen 1 (teacher warm-start): primary={m['eval']['primary']:.3f} "
                f"promoted={m['gate']['promoted']} ({m['gate']['reason']})"
            )
            start = runner.registry.next_generation
        else:
            typer.echo(f"warm-start skipped: run already at gen {start}")
    for g in range(start, end):
        m = runner.run_generation(g)
        typer.echo(
            f"gen {g}: primary={m['eval']['primary']:.3f} "
            f"promoted={m['gate']['promoted']} ({m['gate']['reason']})"
        )


@app.command("push-data")
def push_data(
    repo: str = typer.Argument(..., help="HF dataset repo id, e.g. CraftsMan-Labs/slm-rl-mastermind"),
    run_id: str = typer.Option("default"),
    home: str = typer.Option("./runs"),
    public: bool = typer.Option(False, help="Create the repo public (default private)"),
) -> None:
    """Upload an existing run's per-generation datasets to the HF Hub."""
    from pathlib import Path

    from slm_rl.datagen.hf_push import push_generation
    from slm_rl.orchestrator.paths import RunPaths

    paths = RunPaths(home, run_id)
    gens = sorted(paths.root.glob("generations/gen_*"))
    if not gens:
        typer.secho(f"no generations under {paths.root}", fg=typer.colors.RED)
        raise typer.Exit(1)
    for gen_dir in gens:
        g = int(gen_dir.name.split("_")[1])
        url = push_generation(repo, run_id, g, Path(gen_dir), private=not public)
        typer.echo(f"gen {g} -> {url}")


@app.command("eval")
def eval_cmd(
    game: str = typer.Option("mastermind"),
    agent: str = typer.Option("random", help="llm | random"),
    limit: int = typer.Option(None, help="Cap suite episodes (smoke tests)"),
    model: str = typer.Option(None),
    backend: str = typer.Option(None),
    tier: str = typer.Option(None),
    adapter: str = typer.Option(None, help="LoRA adapter path (llm agent)"),
    pruner: bool = typer.Option(False, "--pruner/--no-pruner", help="Teacher menu pruning (side metric only — never the gate)"),
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

    pruner_obj = None
    if pruner:
        from slm_rl.teachers import make_pruner

        pruner_obj = make_pruner(game_cfg)

    try:
        metrics = run_suite(
            game_cls.eval_suite(), make_agent, game_cls, game_cfg,
            limit=limit, pruner=pruner_obj,
        )
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


@app.command()
def watch(
    run: str = typer.Option(..., help="Run id under runs/"),
    home: str = typer.Option("./runs"),
    port: int = typer.Option(8777),
    host: str = typer.Option("127.0.0.1"),
) -> None:
    """Stream a run's episodes live to a browser (read-only observer)."""
    from slm_rl.orchestrator.paths import RunPaths
    from slm_rl.webui.server import serve

    paths = RunPaths(home, run)
    if not paths.root.exists():
        typer.secho(f"no run at {paths.root}", fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.echo(f"watching {paths.root} -> http://{host}:{port}/")
    serve(paths.root, host=host, port=port)


@app.command("new-game")
def new_game(name: str = typer.Argument(...)) -> None:
    """Scaffold a new game plugin."""
    _todo("Phase 4")


@app.command()
def playground(
    game: str = typer.Option("space-invaders"),
    home: str = typer.Option("./runs"),
    port: int = typer.Option(8780),
    host: str = typer.Option("127.0.0.1"),
) -> None:
    """Workshop UI: tweak knobs / reward code, launch quick CPU experiments,
    compare on a scoreboard (plan 013). Stdlib-only, local by default."""
    from slm_rl.playground.server import serve

    typer.echo(f"playground for {game} -> http://{host}:{port}/?game={game}")
    serve(home, game, host=host, port=port)


if __name__ == "__main__":
    app()
