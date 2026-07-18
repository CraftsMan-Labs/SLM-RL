"""SLM-RL command line."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="slm-rl",
    help="A self-improving game gymnasium for small language models.",
    no_args_is_help=True,
)


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


def _resolve_model_backend(model, backend, tier):
    """Resolve (model_id, backend_name, quantization) from a tier + overrides."""
    from slm_rl.config.loader import load_tiers
    from slm_rl.platform.hardware import resolve_tier

    t = resolve_tier(load_tiers(), forced_name=tier)
    return model or t.model, backend or t.backend, t.quantization


def _build_model_agent(kind, game_cls, game_cfg, model, backend, tier, adapter, temperature):
    """Build llm or vl agent + backend. kind="vl" skips tier resolution
    (demo-only transformers-vl; model defaults to extra.vl_model)."""
    from slm_rl.inference.base import GenParams, create_backend

    if kind == "vl":
        from slm_rl.agents.vl_agent import VLAgent
        from slm_rl.inference.transformers_vl_be import DEFAULT_VL_MODEL

        model_id = model or game_cfg.extra.get("vl_model", DEFAULT_VL_MODEL)
        be = create_backend("transformers-vl", model_id)
        AgentCls = VLAgent
    else:
        from slm_rl.agents.llm_agent import LLMAgent

        model_id, backend_name, quant = _resolve_model_backend(model, backend, tier)
        be = create_backend(backend_name, model_id, quant)
        AgentCls = LLMAgent
    if adapter:
        be.load_adapter(adapter)
    agent = AgentCls(
        be, game_cls(game_cfg).system_prompt(),
        gen_params=GenParams(max_tokens=256, temperature=temperature),
    )
    return agent, be, model_id


@app.command()
def rollout(
    game: str = typer.Option("boxing"),
    agent: str = typer.Option("random", help="llm | vl | random | solver (teacher)"),
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
        llm, backend_obj, model_id = _build_model_agent(
            "llm", game_cls, game_cfg, model, backend, tier, adapter, temperature=0.8
        )
        make_agent = lambda i: llm  # noqa: E731 (stateless per turn -> reuse)
    elif agent == "vl":
        vl, backend_obj, model_id = _build_model_agent(
            "vl", game_cls, game_cfg, model, backend, tier, adapter, temperature=0.8
        )
        make_agent = lambda i: vl  # noqa: E731 (stateless per turn -> reuse)
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
    game: str = typer.Option("boxing"),
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
    early_stop_patience: int = typer.Option(0, help="Stop after N flat eval windows (0=off); --decisions becomes a ceiling"),
    early_stop_min_delta: float = typer.Option(0.02, help="Min eval improvement (fraction of best) that counts as progress"),
) -> None:
    """Train a CleanRL-pattern DQN teacher over Game.vector_obs() (plan 012)."""
    from slm_rl.config.loader import load_game_config
    from slm_rl.teachers.dqn import train_dqn

    game_cfg = load_game_config(game)
    summary = train_dqn(
        game_cfg, decisions=decisions, out_path=out, device=device, seed=seed,
        early_stop_patience=early_stop_patience, early_stop_min_delta=early_stop_min_delta,
    )
    typer.echo(f"summary: {summary}")


@app.command()
def evolve(
    game: str = typer.Option("boxing"),
    generations: int = typer.Option(2),
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
    dataset_url: str = typer.Option(None, help="Public HF dataset pack URL/id — replaces live warm-start"),
    dqn_url: str = typer.Option(None, help="Optional public HF URL/id for dqn.pt"),
    adapter_url: str = typer.Option(
        None,
        help="Public HF *model* repo with adapter/ (published SFT LoRA) — skip re-SFT, start RL from gen 2",
    ),
    skip_baseline: bool = typer.Option(
        False,
        "--skip-baseline",
        help="Skip gen-0 frozen eval (use when a baked pack is ready and you only want SFT)",
    ),
    skip_eval: bool = typer.Option(
        False,
        "--skip-eval",
        help="Skip post-train frozen eval on warm-start/pack gens (SFT-only; still writes adapter)",
    ),
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
        "dataset_url": dataset_url, "dqn_url": dqn_url, "adapter_url": adapter_url,
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
    # CLI flags win over materialized yaml when provided
    if dataset_url:
        cfg.dataset_url = dataset_url
    if dqn_url:
        cfg.dqn_url = dqn_url
    if adapter_url:
        cfg.adapter_url = adapter_url
    runner = GenerationRunner(cfg, config_dir=cfg_dir)
    typer.echo(
        f"[evolve] run_id={cfg.run_id} game={cfg.game} generations={generations} "
        f"backend={runner.backend_name} model={runner.model_id}",
        err=False,
    )
    runner.ensure_baseline(skip=skip_baseline)
    start = runner.registry.next_generation
    end = start + generations
    typer.echo(f"[evolve] next generation={start} (will run until {end - 1})")

    pack_url = (cfg.dataset_url or "").strip() or None
    sft_url = (cfg.adapter_url or "").strip() or None
    if sft_url and start == 1:
        from slm_rl.orchestrator.registry import ModelRegistry
        from slm_rl.packs import import_adapter_as_champion, resolve_adapter

        typer.echo(f"[packs] importing published SFT adapter from {sft_url}")
        src = resolve_adapter(sft_url, cfg.home, cfg.game)
        import_adapter_as_champion(
            runner.paths.root, src, model_id=runner.model_id, game=cfg.game,
        )
        # import wrote registry on disk; reload so next_generation sees gen 1
        runner.registry = ModelRegistry(runner.paths.registry)
        typer.echo(
            "gen 1 (imported SFT adapter): promoted as RL initialization "
            f"(champion={runner.registry.champion})"
        )
        start = runner.registry.next_generation
        # Imported SFT is free (not counted). `generations` = RL steps after it.
        # Without this, gens=1 left range(start,end) empty after import.
        end = start + generations
        if pack_url:
            typer.echo(
                f"[packs] dataset pack {pack_url} noted but unused for gen 1 "
                "(SFT model URL takes priority over re-SFT from demos)"
            )
        typer.echo(f"[evolve] after SFT import: next={start} (will run until {end - 1})")
    elif sft_url and start != 1:
        typer.echo(f"imported adapter skipped: run already at gen {start}")
    elif pack_url and start == 1:
        from slm_rl.packs import materialize_rollouts, resolve_dqn, resolve_pack

        typer.echo(f"[packs] resolving {pack_url}")
        pack = resolve_pack(pack_url, cfg.home, cfg.game)
        materialize_rollouts(pack, runner.paths.rollouts(1))
        dqn = (cfg.dqn_url or "").strip() or None
        if dqn:
            pt = resolve_dqn(dqn, cfg.home, cfg.game)
            cfg.teacher.dqn_checkpoint = str(pt)
        m = runner.run_generation(
            1, teacher=True, skip_rollout=True, skip_eval=skip_eval,
        )
        typer.echo(
            f"gen 1 (baked pack): primary={m['eval'].get('primary', float('nan'))} "
            f"promoted={m['gate']['promoted']} ({m['gate']['reason']})"
        )
        start = runner.registry.next_generation
    elif pack_url and start != 1:
        typer.echo(f"baked pack skipped: run already at gen {start}")
    elif warm_start:
        if start == 1:
            m = runner.run_generation(1, teacher=True, skip_eval=skip_eval)
            typer.echo(
                f"gen 1 (teacher warm-start): primary={m['eval'].get('primary', float('nan'))} "
                f"promoted={m['gate']['promoted']} ({m['gate']['reason']})"
            )
            start = runner.registry.next_generation
        else:
            typer.echo(f"warm-start skipped: run already at gen {start}")
    stop_after = runner.cfg.gate.max_consecutive_failures
    for g in range(start, end):
        m = runner.run_generation(g)
        typer.echo(
            f"gen {g}: primary={m['eval']['primary']:.3f} "
            f"promoted={m['gate']['promoted']} ({m['gate']['reason']})"
        )
        # ponytail: same threshold as LR remediation; reject streak = stop, don't burn the plan
        fails = runner.registry.consecutive_failures
        if not m["gate"]["promoted"] and fails >= stop_after:
            typer.echo(
                f"[evolve] early stop at gen {g}: {fails} consecutive rejects "
                f"(limit {stop_after})",
            )
            break


@app.command("push-data")
def push_data(
    repo: str = typer.Argument(..., help="HF dataset repo id, e.g. BLANK/slm-rl-boxing"),
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
    game: str = typer.Option("boxing"),
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
        llm, backend_obj, _ = _build_model_agent(
            "llm", game_cls, game_cfg, model, backend, tier, adapter, temperature=0.2
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
def watch(
    run: str = typer.Option(..., help="Run id under runs/"),
    home: str = typer.Option("./runs"),
    port: int = typer.Option(8777),
    host: str = typer.Option("127.0.0.1"),
    fps: float = typer.Option(
        30.0,
        help="Frame playback rate for /frames (lower = slower; try 4 to watch a full DQN game)",
    ),
) -> None:
    """Stream a run's episodes live to a browser (read-only observer)."""
    from slm_rl.orchestrator.paths import RunPaths
    from slm_rl.webui.server import serve

    paths = RunPaths(home, run)
    if not paths.root.exists():
        typer.secho(f"no run at {paths.root}", fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.echo(f"watching {paths.root} -> http://{host}:{port}/ (fps={fps})")
    serve(paths.root, host=host, port=port, fps=fps)


@app.command()
def theater(
    run_id: str = typer.Option(..., "--run-id", help="Run id under runs/"),
    game: str = typer.Option(None, help="Defaults to the run's own game"),
    home: str = typer.Option("./runs"),
    episodes: int = typer.Option(10, help="Exhibition episodes per side"),
    seed_start: int = typer.Option(20_000, help="First exhibition seed (>=20000, disjoint from eval/rollout seeds)"),
    config_dir: str = typer.Option(None, help="Alternate configs/ root (playground experiments)"),
) -> None:
    """Play the base model and the run's champion on the SAME seeds, writing
    both sides under `<run_dir>/theater/{base,champion}/` in the exact layout
    the live-play viewer already understands (plan 020) -- no viewer changes
    needed to watch either side."""
    from pathlib import Path

    from slm_rl.orchestrator.paths import RunPaths
    from slm_rl.theater.exhibition import run_exhibition

    paths = RunPaths(home, run_id)
    if not paths.root.exists():
        typer.secho(f"no run at {paths.root}", fg=typer.colors.RED)
        raise typer.Exit(1)
    if not (paths.root / "run_config.yaml").exists():
        # Written by GenerationRunner.__init__ (i.e. by `evolve`), not by
        # plain `rollout` -- a quick-experiment-only run has no champion to
        # exhibit and no frozen config to resolve the model/backend from.
        typer.secho(
            f"no run_config.yaml at {paths.root} -- this run has never "
            "gone through `slm-rl evolve` (theater needs a frozen config "
            "and a registry to resolve the champion)",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    resolved_game = game
    if resolved_game is None:
        import yaml

        run_cfg_data = yaml.safe_load((paths.root / "run_config.yaml").read_text(encoding="utf-8"))
        resolved_game = run_cfg_data["game"]

    cfg_dir = Path(config_dir) if config_dir else None
    result = run_exhibition(
        paths.root, resolved_game, episodes=episodes, seed_start=seed_start, config_dir=cfg_dir,
    )
    typer.echo(f"base exhibition -> {result.base_dir}")
    if result.champion_dir is None:
        typer.secho(result.message, fg=typer.colors.YELLOW)
    else:
        typer.echo(f"champion (gen {result.champion_generation}) exhibition -> {result.champion_dir}")


@app.command("play-again")
def play_again(
    run_id: str = typer.Option(..., "--run-id", help="Run id under runs/"),
    generation: int = typer.Option(..., "--generation", help="Checkpoint gen (0 = base)"),
    game: str = typer.Option(None, help="Defaults to the run's own game"),
    home: str = typer.Option("./runs"),
    episodes: int = typer.Option(10, help="Episodes to play"),
    seed: int = typer.Option(20_000, help="First episode seed"),
    temperature: float = typer.Option(0.2, help="LLM sampling temperature"),
    config_dir: str = typer.Option(None, help="Alternate configs/ root (playground experiments)"),
) -> None:
    """Replay one checkpoint (base or a LoRA gen) into theater/play/ for the
    live viewer (plan 026 Phase G). No training — thin exhibition `_play_side`
    wrapper. Playground `launch_play_again` shells this command."""
    from pathlib import Path

    from slm_rl.orchestrator.paths import RunPaths
    from slm_rl.theater.exhibition import run_play_again

    paths = RunPaths(home, run_id)
    if not paths.root.exists():
        typer.secho(f"no run at {paths.root}", fg=typer.colors.RED)
        raise typer.Exit(1)
    if not (paths.root / "run_config.yaml").exists():
        typer.secho(
            f"no run_config.yaml at {paths.root} — evolve first "
            "(play-again needs a frozen config to resolve model/backend)",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    resolved_game = game
    if resolved_game is None:
        import yaml

        run_cfg_data = yaml.safe_load((paths.root / "run_config.yaml").read_text(encoding="utf-8"))
        resolved_game = run_cfg_data["game"]

    cfg_dir = Path(config_dir) if config_dir else None
    try:
        result = run_play_again(
            paths.root, resolved_game, generation=generation,
            episodes=episodes, seed_start=seed, temperature=temperature,
            config_dir=cfg_dir,
        )
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1) from exc
    typer.echo(f"play-again gen {result.generation} -> {result.play_dir}")


@app.command()
def playground(
    game: str | None = typer.Option(
        None, help="Default game for form preselect / baseline (optional; UI picks per experiment)",
    ),
    home: str = typer.Option("./runs"),
    port: int = typer.Option(8780),
    host: str = typer.Option("127.0.0.1"),
) -> None:
    """Workshop UI: tweak knobs / reward code, launch quick CPU experiments,
    compare on a scoreboard (plan 013). Stdlib-only, local by default.
    Game is chosen in the UI per experiment; --game only sets the default."""
    from slm_rl.games.registry import available_games
    from slm_rl.playground.server import serve

    games = available_games()
    if not games:
        typer.secho("no games registered", fg=typer.colors.RED)
        raise typer.Exit(1)
    if game is None:
        default_game = games[0]
    elif game not in games:
        typer.secho(f"unknown game {game!r}. Available: {', '.join(games)}", fg=typer.colors.RED)
        raise typer.Exit(1)
    else:
        default_game = game
    typer.echo(f"playground (default game: {default_game}) -> http://{host}:{port}/")
    serve(home, default_game, host=host, port=port)


if __name__ == "__main__":
    app()
