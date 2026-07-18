"""GenerationRunner: one generation = ROLLOUT -> DATASET -> TRAIN -> EVAL ->
GATE (promote/rollback). `slm-rl evolve` loops this.

Single-GPU handoff: the inference backend is created and `close()`d around
each phase so rollout, training, and eval never hold the GPU simultaneously.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from slm_rl.agents.llm_agent import LLMAgent
from slm_rl.config.loader import load_game_config, load_tiers
from slm_rl.config.schema import RunConfig
from slm_rl.eval.gate import EvalGate
from slm_rl.eval.suites import run_suite
from slm_rl.games.registry import get_game
from slm_rl.inference.base import GenParams, create_backend
from slm_rl.orchestrator.paths import RunPaths
from slm_rl.orchestrator.registry import ModelRegistry
from slm_rl.platform.hardware import resolve_tier
from slm_rl.rollout.runner import EpisodeRunner
from slm_rl.training.base import create_strategy

ROLLOUT_TEMP = 0.8   # exploration during data collection (D10)
EVAL_TEMP = 0.2      # near-greedy for stable, paired evaluation


def _log(msg: str) -> None:
    """Flushed progress for evolve.log / playground UI (looks stuck without it)."""
    print(msg, flush=True)


class GenerationRunner:
    def __init__(self, cfg: RunConfig, config_dir: Path | None = None):
        self.cfg = cfg
        # Alternate configs/ root (playground experiments, plan 013): the
        # GAME config must come from the same materialized dir as the run
        # config, or an attendee's game-level knobs (monitor thresholds,
        # reward_hook) silently vanish on the evolve path. None = repo
        # configs/ = exact prior behavior. Deliberately NOT threaded into
        # load_tiers: tier resolution is hardware detection, not an
        # experiment knob, and experiment dirs don't carry hardware.yaml.
        self.config_dir = config_dir
        tier = resolve_tier(load_tiers(), forced_name=cfg.tier)
        self.model_id = cfg.model or tier.model
        self.backend_name = cfg.backend or tier.backend
        self.quantization = tier.quantization
        self.strategy_name = cfg.train_strategy or tier.train

        self.game_cls = get_game(cfg.game)
        self.game_cfg = load_game_config(cfg.game, config_dir=config_dir)
        self.suite = self.game_cls.eval_suite()
        self.paths = RunPaths(cfg.home, cfg.run_id)
        self.registry = ModelRegistry(self.paths.registry)
        self.gate = EvalGate(cfg.gate)

        self.pruner = None
        if cfg.teacher.pruner:
            from slm_rl.teachers import make_pruner

            self.pruner = make_pruner(self.game_cfg, top_k=cfg.teacher.pruner_top_k)
        # auto-remediation mutates LR; restored on the next promotion
        self._orig_lr = cfg.train.learning_rate

        self.paths.root.mkdir(parents=True, exist_ok=True)
        frozen = self.paths.root / "run_config.yaml"
        if not frozen.exists():
            import yaml

            frozen.write_text(yaml.safe_dump(cfg.model_dump()))

    # -- helpers ---------------------------------------------------------

    def _backend(self, adapter: Path | None):
        backend = create_backend(self.backend_name, self.model_id, self.quantization)
        if adapter is not None:
            backend.load_adapter(adapter)
        return backend

    def _agent(self, backend, temperature: float) -> LLMAgent:
        params = GenParams(max_tokens=self.cfg.train.max_completion_tokens, temperature=temperature)
        return LLMAgent(backend, self.game_cls(self.game_cfg).system_prompt(), gen_params=params)

    def _batch_size(self) -> int:
        # batching is transformers/vLLM only (plan 005); llama.cpp/MLX (the
        # 8GB default) always stay serial regardless of the configured knob.
        if self.backend_name not in ("transformers", "transformers-4bit"):
            return 1
        return self.cfg.train.rollout_batch_size

    def _eval(
        self,
        adapter: Path | None,
        limit: int | None = None,
        pruner=None,
        *,
        label: str = "eval",
    ) -> dict:
        # pruner is ONLY for the eval_pruned side metric; the gate eval never
        # passes it (steering must not be counted as model improvement).
        # limit: baseline/gate calls omit it, so it defaults to
        # game_cfg.eval_episodes -- a PREFIX of the frozen suite seeds
        # (run_suite's suite.seeds[:limit]), pairing across generations
        # preserved. eval_pruned passes its own eval_pruned_episodes limit
        # explicitly, which overrides this default (not the other way).
        if limit is None:
            limit = self.game_cfg.eval_episodes
        n_eps = min(limit, len(self.suite.seeds))
        who = "base model" if adapter is None else f"adapter={adapter.name}"
        _log(f"[evolve] {label}: loading backend={self.backend_name} model={self.model_id} ({who})")
        backend = self._backend(adapter)
        try:
            agent = self._agent(backend, EVAL_TEMP)
            _log(f"[evolve] {label}: playing {n_eps} frozen-suite episodes ({self.cfg.game})")

            # ponytail: log every episode + every 5th for long Atari suites
            def _progress(done: int, total: int, summary: dict) -> None:
                if done == 1 or done == total or done % 5 == 0:
                    score = summary.get("cum_reward")
                    outcome = summary.get("outcome") or "—"
                    _log(
                        f"[evolve] {label}: episode {done}/{total} "
                        f"outcome={outcome} score={score}"
                    )

            metrics = run_suite(
                self.suite, lambda: agent, self.game_cls, self.game_cfg,
                limit=limit, pruner=pruner,
                batch_size=self._batch_size(), backend=backend,
                system_prompt=agent.system_prompt, gen_params=agent.params,
                on_episode=_progress,
            )
            _log(
                f"[evolve] {label}: done primary={metrics['primary']:.3f} "
                f"({self.suite.primary_metric})"
            )
            return metrics
        finally:
            backend.close()

    def _champion_metrics(self) -> dict:
        from slm_rl.eval.gate import coerce_gate_metrics

        raw = json.loads(
            (self.paths.generation(self.registry.champion) / "eval" / "results.json").read_text()
        )
        # Guard: old workshop stubs may omit rate keys — coerce before gate.
        return coerce_gate_metrics(raw)

    def _write_json(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    # -- public API ------------------------------------------------------

    def ensure_baseline(self, *, skip: bool = False) -> dict:
        """Evaluate gen 0 (base model, no adapter) once and cache its metrics.

        ``skip=True`` writes a stub so pack/SFT warm-starts can train without
        burning wall-clock on the frozen suite (operator already has demos).
        """
        results = self.paths.generation(0) / "eval" / "results.json"
        if results.exists():
            _log("[evolve] gen 0 baseline: cached — skipping re-eval")
            return json.loads(results.read_text())
        if skip:
            from slm_rl.eval.gate import stub_eval_metrics

            metrics = stub_eval_metrics(
                self.cfg.gate.stub_primary,
                note="baseline skipped (--skip-baseline); not comparable to gated runs",
            )
            self._write_json(results, metrics)
            _log(f"[evolve] gen 0 baseline: SKIPPED — stub wrote {results}")
            return metrics
        _log("[evolve] gen 0 baseline: starting (no training yet — measuring stock model)")
        metrics = self._eval(adapter=None, label="gen 0 baseline")
        self._write_json(results, metrics)
        _log(f"[evolve] gen 0 baseline: wrote {results}")
        return metrics

    def run_generation(
        self,
        generation: int,
        teacher: bool = False,
        *,
        skip_rollout: bool = False,
        skip_eval: bool = False,
    ) -> dict:
        champ = self.registry.champion
        champ_adapter = self.paths.adapter(champ) if champ > 0 else None
        champ_metrics = self._champion_metrics()
        kind = (
            "baked-pack warm-start" if skip_rollout
            else "teacher warm-start" if teacher
            else "RL"
        )
        _log(f"[evolve] gen {generation}: start ({kind}), champion=gen {champ}")

        # 1. ROLLOUT (teacher plays the warm-start generation, else the
        # champion explores). Teachers never touch eval — only rollout.
        # skip_rollout: workshop bake pack already materialized into rollouts/.
        if skip_rollout:
            backend = None
            model_id = "teacher:baked_pack"
            strategy_name = "reject_sft"
            n, wins = _count_rollout_episodes(self.paths.rollouts(generation))
            if n == 0:
                raise FileNotFoundError(
                    f"no baked rollouts at {self.paths.rollouts(generation)}"
                )
            _log(f"[evolve] gen {generation}: using baked pack ({n} episodes)")
        elif teacher:
            from slm_rl.teachers import make_teacher

            backend = None
            agent, model_id = make_teacher(
                self.game_cfg, seed=self.cfg.seed, dqn_checkpoint=self.cfg.teacher.dqn_checkpoint,
            )
            n = self.cfg.teacher.warmstart_episodes
            strategy_name = "reject_sft"  # expert traces distill via SFT on every tier
            _log(f"[evolve] gen {generation}: teacher rollout {n} episodes ({model_id})")
        else:
            _log(f"[evolve] gen {generation}: loading champion for rollout")
            backend = self._backend(champ_adapter)
            agent = self._agent(backend, ROLLOUT_TEMP)
            model_id = self.model_id
            n = self.cfg.train.episodes_per_generation
            strategy_name = self.strategy_name
            _log(f"[evolve] gen {generation}: model rollout {n} episodes")

        if not skip_rollout:
            rollouts = self.paths.rollouts(generation) / f"{self.cfg.game}.jsonl"
            from slm_rl.datagen.writer import RolloutWriter

            # ponytail: 0.1-granularity alternation; pruned and format-mode
            # prompts must both appear (the gate eval is format-mode)
            pruned_lot = round(self.cfg.teacher.pruner_fraction * 10)
            wins = 0
            finished = 0
            rollout_batch_size = 1 if teacher else self._batch_size()  # teachers are engine-speed; never batch
            seeds = [self.cfg.seed + generation * n + i for i in range(n)]  # disjoint from eval seeds (>=10000)
            pruners = [self.pruner if self.pruner and i % 10 < pruned_lot else None for i in range(n)]
            with RolloutWriter(rollouts) as writer:
                if rollout_batch_size > 1:
                    from slm_rl.rollout.batch_runner import BatchedEpisodeRunner

                    for start in range(0, n, rollout_batch_size):
                        chunk = range(start, min(start + rollout_batch_size, n))
                        runner = BatchedEpisodeRunner(
                            games=[self.game_cls(self.game_cfg) for _ in chunk],
                            seeds=[seeds[i] for i in chunk],
                            episode_ids=[f"g{generation}-{seeds[i]}" for i in chunk],
                            game_cfg=self.game_cfg,
                            backend=backend,
                            system_prompt=agent.system_prompt,
                            gen_params=agent.params,
                            writer=writer,
                            run_id=self.cfg.run_id, generation=generation, model_id=model_id,
                            adapter_ref=str(champ_adapter) if champ_adapter else None,
                            pruners=[pruners[i] for i in chunk],
                        )
                        for summary in runner.run():
                            wins += summary["outcome"] == "win"
                            finished += 1
                            if finished == 1 or finished == n or finished % 10 == 0:
                                _log(
                                    f"[evolve] gen {generation}: rollout "
                                    f"{finished}/{n} outcome={summary.get('outcome') or '—'}"
                                )
                else:
                    for i in range(n):
                        runner = EpisodeRunner(
                            self.game_cls(self.game_cfg), agent, self.game_cfg, writer=writer,
                            run_id=self.cfg.run_id, generation=generation, model_id=model_id,
                            adapter_ref=str(champ_adapter) if champ_adapter else None,
                            pruner=pruners[i],
                        )
                        summary = runner.run_episode(seeds[i], episode_id=f"g{generation}-{seeds[i]}")
                        wins += summary["outcome"] == "win"
                        finished += 1
                        if finished == 1 or finished == n or finished % 10 == 0:
                            _log(
                                f"[evolve] gen {generation}: rollout "
                                f"{finished}/{n} outcome={summary.get('outcome') or '—'}"
                            )
            if backend is not None:
                backend.close()  # 2. free GPU before training
            _log(f"[evolve] gen {generation}: rollout done win_rate={wins}/{n}")

        # 3. DATASET
        from slm_rl.datagen.consolidate import consolidate

        _log(f"[evolve] gen {generation}: consolidating dataset")
        dataset = self.paths.dataset(generation)
        consolidate(self.paths.rollouts(generation), dataset)

        # replay window: this gen + up to N-1 previous gens that have rollouts
        # (plan 004 — cross-generation replay). Gen 0 is baseline-only (no
        # rollouts) and is naturally excluded by the exists() filter; teacher
        # rollouts (gen 1) enter the window like any other generation's.
        window = range(max(1, generation - self.cfg.train.replay_generations + 1), generation + 1)
        sources = [(g, self.paths.rollouts(g)) for g in window if self.paths.rollouts(g).exists()]
        train_view = dataset
        if len(sources) > 1:
            replay_src = self.paths.generation(generation) / "dataset" / "replay_src"
            replay_src.mkdir(parents=True, exist_ok=True)
            for g, src_dir in sources:
                for jsonl in sorted(src_dir.glob("*.jsonl")):
                    link = replay_src / f"g{g:03d}-{jsonl.name}"
                    # is_symlink (not exists): a stale link to a deleted target
                    # must not crash the resume path with FileExistsError
                    if not link.is_symlink():
                        os.symlink(jsonl.resolve(), link)
            train_view = self.paths.generation(generation) / "dataset" / "replay.parquet"
            consolidate(replay_src, train_view)

        # 4. TRAIN
        _log(f"[evolve] gen {generation}: train start strategy={strategy_name}")
        strategy = create_strategy(strategy_name, self.cfg.train, self.model_id, self.game_cfg)
        result = strategy.train(train_view, self.paths.generation(generation), init_adapter=champ_adapter)
        _log(f"[evolve] gen {generation}: train done metrics={result.metrics}")

        # 5. EVAL candidate + 6. GATE
        if result.metrics.get("entropy_collapsed"):
            # force-reject without an eval pass: the policy is degenerate
            promote, reason = False, "train entropy collapsed"
            cand_metrics = champ_metrics
            _log(f"[evolve] gen {generation}: gate skip — {reason}")
        elif result.adapter_path is None:  # nothing trained (no data)
            promote, reason = False, "no training data this generation"
            cand_metrics = champ_metrics
            _log(f"[evolve] gen {generation}: gate skip — {reason}")
        elif teacher or skip_rollout:
            # SFT warm-start is initialization, not a candidate (D12): the
            # gate is a win-rate margin check, which is noisy at the
            # warm-start's win-rate scale and discarded a 35%->0% invalid-rate
            # improvement twice (Jul 2026). Eval stays honest; only the
            # promotion decision is unconditional. Baked packs use the same
            # ungated adopt (workshop day-of path).
            if skip_eval:
                cand_metrics = {
                    **champ_metrics,
                    "skipped": True,
                    "note": "post-train eval skipped (--skip-eval)",
                }
                _log(
                    f"[evolve] gen {generation}: post-train eval SKIPPED "
                    "(SFT-only path)"
                )
            else:
                cand_metrics = self._eval(
                    result.adapter_path, label=f"gen {generation} post-train eval",
                )
            promote = True
            reason = (
                "baked pack adopted as RL initialization (not gated)"
                if skip_rollout
                else "teacher warm-start adopted as RL initialization (not gated)"
            )
        else:
            cand_metrics = self._eval(
                result.adapter_path, label=f"gen {generation} gate eval",
            )
            promote, reason = self.gate.decide(champ_metrics, cand_metrics)

        if promote:
            self.registry.promote(generation, reason)
            self._write_json(self.paths.generation(generation) / "eval" / "results.json", cand_metrics)
            # remediation is a crutch, not a new baseline: a promotion clears it
            self.cfg.train.learning_rate = self._orig_lr
            _log(f"[evolve] gen {generation}: PROMOTED — {reason}")
        else:
            self.registry.reject(generation, reason)
            _log(f"[evolve] gen {generation}: rejected — {reason}")

        # 7. Persist metrics + manifest
        metrics = {
            "rollout": {"episodes": n, "train_win_rate": wins / n if n else 0.0},
            "train": result.metrics,
            "eval": cand_metrics,
            "gate": {"promoted": promote, "reason": reason},
        }
        # side metric: candidate + pruner (the product view). Small suite,
        # recorded only — the gate above never saw it.
        if (
            self.pruner is not None
            and self.cfg.teacher.eval_pruned_episodes > 0
            and result.adapter_path is not None
            and not result.metrics.get("entropy_collapsed")
        ):
            metrics["eval_pruned"] = self._eval(
                result.adapter_path,
                limit=self.cfg.teacher.eval_pruned_episodes,
                pruner=self.pruner,
            )
        self._write_json(self.paths.metrics(generation), metrics)
        self._write_json(self.paths.manifest(generation), {
            "base_model": self.model_id,
            "backend": self.backend_name,
            "strategy": strategy_name,
            "rollout_model": model_id,  # "teacher:..." on the warm-start gen
            "parent_generation": champ,
            "adapter": "adapter/",
            "config_hash": hashlib.sha256(json.dumps(self.cfg.model_dump(), sort_keys=True, default=str).encode()).hexdigest()[:16],
            "git_sha": _git_sha(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # 8. Optional dataset publication (best-effort, never kills the loop)
        if self.cfg.hf_dataset_repo:
            from slm_rl.datagen.hf_push import try_push_generation

            url = try_push_generation(
                self.cfg.hf_dataset_repo, self.cfg.run_id, generation,
                self.paths.generation(generation),
            )
            if url:
                print(f"datasets pushed: {url}")

        # 9. Auto-remediation after repeated failures: LR only, floored.
        # entropy_bonus escalation removed — a single doubling (0.01 -> 0.02)
        # sent train entropy to 7.82 (random play, Jul 2026); collapse is
        # already guarded by EntropyFloorCallback + the gate's entropy floor.
        if self.registry.consecutive_failures >= self.cfg.gate.max_consecutive_failures:
            self.cfg.train.learning_rate = max(self.cfg.train.learning_rate / 2, 1e-6)

        return metrics


def _count_rollout_episodes(rollout_dir: Path) -> tuple[int, int]:
    """(n_episodes, n_wins) from jsonl; last row per episode_id wins."""
    outcomes: dict[str, str | None] = {}
    if not rollout_dir.is_dir():
        return 0, 0
    for jsonl in sorted(rollout_dir.glob("*.jsonl")):
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                outcomes[row["episode_id"]] = row.get("outcome")
    wins = sum(1 for o in outcomes.values() if o == "win")
    return len(outcomes), wins


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except Exception:
        return None
