"""EvalGate: checkpoint gating — a new generation is promoted only if it
beats the champion on the frozen suite without regressing on doom-loop,
invalid-action, or entropy criteria (D4, training level). Rollback is simply
never moving the champion pointer."""

from __future__ import annotations

from slm_rl.config.schema import GateConfig


class EvalGate:
    def __init__(self, cfg: GateConfig):
        self.cfg = cfg

    def decide(self, champion_metrics: dict, candidate_metrics: dict) -> tuple[bool, str]:
        """Returns (promote, reason)."""
        raise NotImplementedError("Phase 1")
