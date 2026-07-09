"""GenerationRunner: one generation = ROLLOUT -> DATASET -> TRAIN -> EVAL ->
GATE (promote/rollback). `slm-rl evolve` loops this. Auto-remediation after
repeated gate failures: halve LR, raise entropy bonus, optionally run the
antidoom hygiene stage."""

from __future__ import annotations

from slm_rl.config.schema import RunConfig


class GenerationRunner:
    def __init__(self, cfg: RunConfig):
        raise NotImplementedError("Phase 1")

    def run_generation(self, generation: int) -> dict:
        """Executes one full cycle; returns the generation's metrics."""
        raise NotImplementedError("Phase 1")
