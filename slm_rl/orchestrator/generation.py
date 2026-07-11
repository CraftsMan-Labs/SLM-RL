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


class GenerationRunner:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        tier = resolve_tier(load_tiers(), forced_name=cfg.tier)
        self.model_id = cfg.model or tier.model
        self.backend_name = cfg.backend or tier.backend
        self.quantization = tier.quantization
        self.strategy_name = cfg.train_strategy or tier.train

        self.game_cls = get_game(cfg.game)
        self.game_cfg = load_game_config(cfg.game)
        self.suite = self.game_cls.eval_suite()
        self.paths = RunPaths(cfg.home, cfg.run_id)
        self.registry = ModelRegistry(self.paths.registry)
        self.gate = EvalGate(cfg.gate)

        self.pruner = None
        if cfg.teacher.pruner:
            from slm_rl.teachers import make_pruner

            self.pruner = make_pruner(self.game_cfg, top_k=cfg.teacher.pruner_top_k)
        # auto-remediation mutates these; restored on the next promotion
        self._orig_lr = cfg.train.learning_rate
        self._orig_entropy_bonus = cfg.train.entropy_bonus

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
        if self.backend_name not in ("transformers", "transformers-4bit", "vllm"):
            return 1
        return self.cfg.train.rollout_batch_size

    def _eval(self, adapter: Path | None, limit: int | None = None, pruner=None) -> dict:
        # pruner is ONLY for the eval_pruned side metric; the gate eval never
        # passes it (steering must not be counted as model improvement)
        backend = self._backend(adapter)
        try:
            agent = self._agent(backend, EVAL_TEMP)
            return run_suite(
                self.suite, lambda: agent, self.game_cls, self.game_cfg,
                limit=limit, pruner=pruner,
                batch_size=self._batch_size(), backend=backend,
                system_prompt=agent.system_prompt, gen_params=agent.params,
            )
        finally:
            backend.close()

    def _champion_metrics(self) -> dict:
        return json.loads((self.paths.generation(self.registry.champion) / "eval" / "results.json").read_text())

    def _write_json(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    # -- public API ------------------------------------------------------

    def ensure_baseline(self) -> dict:
        """Evaluate gen 0 (base model, no adapter) once and cache its metrics."""
        results = self.paths.generation(0) / "eval" / "results.json"
        if results.exists():
            return json.loads(results.read_text())
        metrics = self._eval(adapter=None)
        self._write_json(results, metrics)
        return metrics

    def run_generation(self, generation: int, teacher: bool = False) -> dict:
        champ = self.registry.champion
        champ_adapter = self.paths.adapter(champ) if champ > 0 else None
        champ_metrics = self._champion_metrics()

        # 1. ROLLOUT (teacher plays the warm-start generation, else the
        # champion explores). Teachers never touch eval — only rollout.
        if teacher:
            from slm_rl.teachers import make_teacher

            backend = None
            agent, model_id = make_teacher(self.game_cfg, seed=self.cfg.seed)
            n = self.cfg.teacher.warmstart_episodes
            strategy_name = "reject_sft"  # expert traces distill via SFT on every tier
        else:
            backend = self._backend(champ_adapter)
            agent = self._agent(backend, ROLLOUT_TEMP)
            model_id = self.model_id
            n = self.cfg.train.episodes_per_generation
            strategy_name = self.strategy_name
        rollouts = self.paths.rollouts(generation) / f"{self.cfg.game}.jsonl"
        from slm_rl.datagen.writer import RolloutWriter

        # ponytail: 0.1-granularity alternation; pruned and format-mode
        # prompts must both appear (the gate eval is format-mode)
        pruned_lot = round(self.cfg.teacher.pruner_fraction * 10)
        wins = 0
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
        if backend is not None:
            backend.close()  # 2. free GPU before training

        # 3. DATASET
        from slm_rl.datagen.consolidate import consolidate

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
        strategy = create_strategy(strategy_name, self.cfg.train, self.model_id, self.game_cfg)
        result = strategy.train(train_view, self.paths.generation(generation), init_adapter=champ_adapter)

        # 5. EVAL candidate + 6. GATE
        if result.metrics.get("entropy_collapsed"):
            # force-reject without an eval pass: the policy is degenerate
            promote, reason = False, "train entropy collapsed"
            cand_metrics = champ_metrics
        elif result.adapter_path is None:  # nothing trained (no data)
            promote, reason = False, "no training data this generation"
            cand_metrics = champ_metrics
        elif teacher:
            # SFT warm-start is initialization, not a candidate (D12): the
            # gate is a win-rate margin check, which is noisy at the
            # warm-start's win-rate scale and discarded a 35%->0% invalid-rate
            # improvement twice (Jul 2026). Eval stays honest; only the
            # promotion decision is unconditional.
            cand_metrics = self._eval(result.adapter_path)
            promote = True
            reason = "teacher warm-start adopted as RL initialization (not gated)"
        else:
            cand_metrics = self._eval(result.adapter_path)
            promote, reason = self.gate.decide(champ_metrics, cand_metrics)

        if promote:
            self.registry.promote(generation, reason)
            self._write_json(self.paths.generation(generation) / "eval" / "results.json", cand_metrics)
            # remediation is a crutch, not a new baseline: a promotion clears it
            self.cfg.train.learning_rate = self._orig_lr
            self.cfg.train.entropy_bonus = self._orig_entropy_bonus
        else:
            self.registry.reject(generation, reason)

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


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except Exception:
        return None
