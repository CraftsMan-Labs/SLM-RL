"""Exhibition runner: plays the SAME seeds with the base model and the
champion adapter, back to back, and writes both sides in the exact on-disk
layout the existing viewer stack (`slm_rl/webui/{tailer,replay}.py`) already
globs — `<run_dir>/theater/<side>/generations/gen_000/rollouts/<game>.jsonl`.
That layout match is the whole trick (plan 020 Context point 2): no viewer
code has to change to "support" theater dirs, because a theater dir already
looks like any other run's gen_000.

8GB rule (Hard rule 2): one model resident at a time. `run_exhibition` plays
every base episode, closes that backend, THEN creates the champion backend
(base weights + adapter hot-swapped) and plays the champion episodes. It
never holds two backends open simultaneously.

Seed disjointness (Hard rule 1): exhibition seeds are `20_000 + i`, clear of
both the eval suite's seeds (`eval_seeds_start`, default 10_000, capped well
under 20_000 in every shipped game config) and the rollout formula
`cfg.seed + generation*n + i` (bounded by `episodes_per_generation *
generations`, far below 20_000 for any config shipped in this repo). Document
any game config that could ever collide before raising the base.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from slm_rl.inference.base import create_backend

SEED_START = 20_000


@dataclass
class ExhibitionResult:
    base_dir: Path
    champion_dir: Path | None  # None when the run has no promoted champion
    champion_generation: int
    message: str | None  # set when champion_dir is None (base-only exhibition)


def _write_status(theater_root: Path, **fields: Any) -> None:
    """Heartbeat for the A/B UI (phase / progress / errors). Best-effort."""
    theater_root.mkdir(parents=True, exist_ok=True)
    path = theater_root / "status.json"
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
    data.update(fields)
    data["ts"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _play_side(
    *,
    game_cls,
    game_cfg,
    backend,
    run_id: str,
    generation: int,
    model_id: str,
    adapter_ref: str | None,
    game_name: str,
    episodes: int,
    seed_start: int,
    side_dir: Path,
    temperature: float = 0.2,
    episode_id_prefix: str = "theater",
    max_tokens: int = 24,
    on_episode: Callable[[int, int], None] | None = None,
) -> None:
    from slm_rl.agents.llm_agent import LLMAgent
    from slm_rl.datagen.writer import RolloutWriter
    from slm_rl.inference.base import GenParams
    from slm_rl.rollout.runner import EpisodeRunner

    # Re-run overwrites, never appends (the v4 partial-rollout lesson, plan
    # 020 design decision 1): delete the whole side dir up front so a stale
    # rollout file from a previous exhibition can never be read alongside
    # fresh records.
    if side_dir.exists():
        shutil.rmtree(side_dir)
    rollouts_dir = side_dir / "generations" / f"gen_{generation:03d}" / "rollouts"
    out_path = rollouts_dir / f"{game_name}.jsonl"

    params = GenParams(max_tokens=max_tokens, temperature=temperature)
    agent = LLMAgent(backend, game_cls(game_cfg).system_prompt(), gen_params=params)

    with RolloutWriter(out_path) as writer:
        for i in range(episodes):
            seed = seed_start + i
            if on_episode is not None:
                on_episode(i + 1, episodes)
            runner = EpisodeRunner(
                game_cls(game_cfg), agent, game_cfg, writer=writer,
                run_id=run_id, generation=generation, model_id=model_id,
                adapter_ref=adapter_ref,
            )
            print(
                f"[theater] gen {generation}: episode {i + 1}/{episodes} seed={seed}",
                flush=True,
            )
            summary = runner.run_episode(seed, episode_id=f"{episode_id_prefix}-{generation}-{seed}")
            print(
                f"[theater] gen {generation}: episode {i + 1}/{episodes} done "
                f"outcome={(summary or {}).get('outcome')}",
                flush=True,
            )


def run_exhibition(
    run_dir: Path,
    game: str,
    *,
    episodes: int = 10,
    seed_start: int = SEED_START,
    config_dir: Path | None = None,
) -> ExhibitionResult:
    """Resolve the run's model/backend/champion from its own `run_config.yaml`
    + `registry.json`, then play `episodes` seeded episodes per side.

    `run_dir` is the run's root (`RunPaths(home, run_id).root`), same as any
    other `runs/<run_id>/`. `config_dir` mirrors the playground's alternate
    configs/ root (plan 013) — the exhibition must read the SAME game config
    an experiment's rollouts used, or its knobs (max_turns, reward_hook, ...)
    would silently diverge from what's being demonstrated.
    """
    import yaml

    from slm_rl.config.loader import load_game_config, load_tiers
    from slm_rl.config.schema import RunConfig
    from slm_rl.games.registry import get_game
    from slm_rl.orchestrator.paths import RunPaths
    from slm_rl.orchestrator.registry import ModelRegistry
    from slm_rl.platform.hardware import resolve_tier

    run_dir = Path(run_dir)
    run_config_path = run_dir / "run_config.yaml"
    cfg = RunConfig(**yaml.safe_load(run_config_path.read_text(encoding="utf-8")))

    tier = resolve_tier(load_tiers(), forced_name=cfg.tier)
    model_id = cfg.model or tier.model
    backend_name = cfg.backend or tier.backend
    quantization = tier.quantization

    game_cls = get_game(game)
    game_cfg = load_game_config(game, config_dir=config_dir)
    max_tokens = int(getattr(cfg.train, "max_completion_tokens", None) or 24)

    registry = ModelRegistry(run_dir / "registry.json")
    champ_gen = registry.champion

    theater_root = run_dir / "theater"
    base_dir = theater_root / "base"
    paths = RunPaths(cfg.home, cfg.run_id)

    def _exhibit(
        generation: int,
        adapter_path: Path | None,
        side_dir: Path,
        *,
        phase: str,
    ) -> None:
        backend = create_backend(backend_name, model_id, quantization)
        try:
            if adapter_path is not None:
                backend.load_adapter(adapter_path)

            def _progress(done: int, total: int) -> None:
                _write_status(
                    theater_root,
                    phase=phase,
                    side=phase,
                    generation=generation,
                    completed=done - 1,
                    episode=done,
                    episodes=total,
                    error=None,
                )

            _play_side(
                game_cls=game_cls, game_cfg=game_cfg, backend=backend,
                run_id=f"{cfg.run_id}-theater", generation=generation, model_id=model_id,
                adapter_ref=str(adapter_path) if adapter_path else None,
                game_name=game, episodes=episodes,
                seed_start=seed_start, side_dir=side_dir,
                max_tokens=max_tokens,
                on_episode=_progress,
            )
        finally:
            backend.close()  # release before next side (8GB rule)

    try:
        # 1. BASE — always played, generation 0, no adapter.
        print(
            f"[theater] base side: {episodes} episodes "
            f"(champion waits until base finishes)",
            flush=True,
        )
        _write_status(
            theater_root,
            phase="base",
            side="base",
            generation=0,
            completed=0,
            episode=0,
            episodes=episodes,
            champion_generation=champ_gen,
            error=None,
            message=None,
        )
        _exhibit(0, None, base_dir, phase="base")
        _write_status(
            theater_root,
            phase="base_done",
            side="base",
            completed=episodes,
            episode=episodes,
            episodes=episodes,
        )

        if champ_gen <= 0:
            # No promotion has ever happened on this run — there is nothing to
            # contrast the base model against. Base-only exhibition, clear
            # message, no synthetic champion dir (an empty/absent viewer run_dir
            # 404s honestly instead of silently showing an empty page).
            msg = (
                f"run {cfg.run_id!r} has no promoted champion yet "
                f"(registry champion={champ_gen}); base-only exhibition."
            )
            _write_status(
                theater_root,
                phase="done",
                side="base",
                message=msg,
                error=None,
            )
            return ExhibitionResult(
                base_dir=base_dir, champion_dir=None, champion_generation=champ_gen,
                message=msg,
            )

        adapter_path = paths.adapter(champ_gen)
        if not adapter_path.is_dir():
            msg = (
                f"champion gen {champ_gen} has no adapter at {adapter_path}; "
                "base-only exhibition."
            )
            print(f"[theater] {msg}", flush=True)
            _write_status(
                theater_root,
                phase="done",
                side="base",
                message=msg,
                error=msg,
            )
            return ExhibitionResult(
                base_dir=base_dir, champion_dir=None, champion_generation=champ_gen,
                message=msg,
            )

        # 2. CHAMPION — generation = champ_gen, adapter hot-swapped onto the
        # SAME base weights. Backend created fresh (rule 2: base backend above
        # is already closed by this point).
        champion_dir = theater_root / "champion"
        print(f"[theater] champion side: gen {champ_gen}, {episodes} episodes", flush=True)
        _write_status(
            theater_root,
            phase="champion",
            side="champion",
            generation=champ_gen,
            completed=0,
            episode=0,
            episodes=episodes,
            error=None,
        )
        _exhibit(champ_gen, adapter_path, champion_dir, phase="champion")
        _write_status(
            theater_root,
            phase="done",
            side="champion",
            completed=episodes,
            episode=episodes,
            episodes=episodes,
            champion_generation=champ_gen,
            error=None,
        )

        return ExhibitionResult(
            base_dir=base_dir, champion_dir=champion_dir,
            champion_generation=champ_gen, message=None,
        )
    except BaseException as exc:
        # KeyboardInterrupt / SIGTERM / OOM mid-base left champion "waiting…"
        # forever in the UI — stamp the failure so Start Theater is obvious.
        err = f"{type(exc).__name__}: {exc}"
        print(f"[theater] FAILED: {err}", flush=True)
        _write_status(
            theater_root,
            phase="failed",
            error=err,
            message=(
                "Theater stopped before champion finished. "
                "Click Start Theater to retry (avoid editing slm_rl/ during a run — "
                "Docker watchfiles restarts kill the job)."
            ),
        )
        raise


@dataclass
class PlayAgainResult:
    play_dir: Path
    generation: int


def run_play_again(
    run_dir: Path,
    game: str,
    *,
    generation: int,
    episodes: int = 10,
    seed_start: int = SEED_START,
    temperature: float = 0.2,
    config_dir: Path | None = None,
) -> PlayAgainResult:
    """Play one checkpoint (base gen 0 or a LoRA adapter) into
    `<run_dir>/theater/play/` — same viewer layout as exhibition sides
    (plan 026 Phase G). No training; reuses `_play_side` only.

    `generation` is resolved by the caller (explicit gen or registry
    champion). gen 0 = base weights, no adapter.
    """
    import yaml

    from slm_rl.config.loader import load_game_config, load_tiers
    from slm_rl.config.schema import RunConfig
    from slm_rl.games.registry import get_game
    from slm_rl.orchestrator.paths import RunPaths
    from slm_rl.platform.hardware import resolve_tier

    run_dir = Path(run_dir)
    cfg = RunConfig(**yaml.safe_load((run_dir / "run_config.yaml").read_text(encoding="utf-8")))

    tier = resolve_tier(load_tiers(), forced_name=cfg.tier)
    model_id = cfg.model or tier.model
    backend_name = cfg.backend or tier.backend
    quantization = tier.quantization

    game_cls = get_game(game)
    game_cfg = load_game_config(game, config_dir=config_dir)
    paths = RunPaths(cfg.home, cfg.run_id)

    adapter_path: Path | None = None
    if generation > 0:
        adapter_path = paths.adapter(generation)
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"no adapter at {adapter_path} for generation {generation}"
            )

    play_dir = run_dir / "theater" / "play"
    max_tokens = int(getattr(cfg.train, "max_completion_tokens", None) or 24)
    backend = create_backend(backend_name, model_id, quantization)
    try:
        if adapter_path is not None:
            backend.load_adapter(adapter_path)
        _play_side(
            game_cls=game_cls, game_cfg=game_cfg, backend=backend,
            run_id=f"{cfg.run_id}-play", generation=generation, model_id=model_id,
            adapter_ref=str(adapter_path) if adapter_path else None,
            game_name=game, episodes=episodes, seed_start=seed_start,
            side_dir=play_dir, temperature=temperature, episode_id_prefix="play",
            max_tokens=max_tokens,
        )
    finally:
        backend.close()

    return PlayAgainResult(play_dir=play_dir, generation=generation)
