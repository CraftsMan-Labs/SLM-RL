"""GenerationRunner: one generation = ROLLOUT -> DATASET -> TRAIN -> EVAL ->
GATE (promote/rollback). `slm-rl evolve` loops this.

Single-GPU handoff: the inference backend is created and `close()`d around
each phase so rollout, training, and eval never hold the GPU simultaneously.
"""

from __future__ import annotations

import hashlib
import json
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

    def _eval(self, adapter: Path | None, limit: int | None = None) -> dict:
        backend = self._backend(adapter)
        try:
            agent = self._agent(backend, EVAL_TEMP)
            return run_suite(self.suite, lambda: agent, self.game_cls, self.game_cfg, limit=limit)
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

    def run_generation(self, generation: int) -> dict:
        champ = self.registry.champion
        champ_adapter = self.paths.adapter(champ) if champ > 0 else None
        champ_metrics = self._champion_metrics()
        n = self.cfg.train.episodes_per_generation

        # 1. ROLLOUT (champion plays, exploring)
        backend = self._backend(champ_adapter)
        agent = self._agent(backend, ROLLOUT_TEMP)
        rollouts = self.paths.rollouts(generation) / f"{self.cfg.game}.jsonl"
        from slm_rl.datagen.writer import RolloutWriter

        wins = 0
        with RolloutWriter(rollouts) as writer:
            for i in range(n):
                seed = self.cfg.seed + generation * n + i  # disjoint from eval seeds (>=10000)
                runner = EpisodeRunner(
                    self.game_cls(self.game_cfg), agent, self.game_cfg, writer=writer,
                    run_id=self.cfg.run_id, generation=generation, model_id=self.model_id,
                    adapter_ref=str(champ_adapter) if champ_adapter else None,
                )
                summary = runner.run_episode(seed, episode_id=f"g{generation}-{seed}")
                wins += summary["outcome"] == "win"
        backend.close()  # 2. free GPU before training

        # 3. DATASET
        from slm_rl.datagen.consolidate import consolidate

        dataset = self.paths.dataset(generation)
        consolidate(self.paths.rollouts(generation), dataset)

        # 4. TRAIN
        strategy = create_strategy(self.strategy_name, self.cfg.train, self.model_id, self.game_cfg)
        result = strategy.train(dataset, self.paths.generation(generation), init_adapter=champ_adapter)

        # 5. EVAL candidate + 6. GATE
        if result.metrics.get("entropy_collapsed"):
            # force-reject without an eval pass: the policy is degenerate
            promote, reason = False, "train entropy collapsed"
            cand_metrics = champ_metrics
        elif result.adapter_path is None:  # nothing trained (no data)
            promote, reason = False, "no training data this generation"
            cand_metrics = champ_metrics
        else:
            cand_metrics = self._eval(result.adapter_path)
            promote, reason = self.gate.decide(champ_metrics, cand_metrics)

        if promote:
            self.registry.promote(generation, reason)
            self._write_json(self.paths.generation(generation) / "eval" / "results.json", cand_metrics)
        else:
            self.registry.reject(generation, reason)

        # 7. Persist metrics + manifest
        metrics = {
            "rollout": {"episodes": n, "train_win_rate": wins / n if n else 0.0},
            "train": result.metrics,
            "eval": cand_metrics,
            "gate": {"promoted": promote, "reason": reason},
        }
        self._write_json(self.paths.metrics(generation), metrics)
        self._write_json(self.paths.manifest(generation), {
            "base_model": self.model_id,
            "backend": self.backend_name,
            "strategy": self.strategy_name,
            "parent_generation": champ,
            "adapter": "adapter/",
            "config_hash": hashlib.sha256(json.dumps(self.cfg.model_dump(), sort_keys=True, default=str).encode()).hexdigest()[:16],
            "git_sha": _git_sha(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # 8. Auto-remediation after repeated failures
        if self.registry.consecutive_failures >= self.cfg.gate.max_consecutive_failures:
            self.cfg.train.learning_rate /= 2
            self.cfg.train.entropy_bonus *= 2

        return metrics


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except Exception:
        return None
