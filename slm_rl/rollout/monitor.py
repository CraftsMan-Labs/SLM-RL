"""DoomLoopMonitor: rollout-level anti-doom-loop machinery (D4).

Signals per step: identical-action streaks, action n-gram loops, state-hash
revisits, reward stagnation, invalid-action streaks. Interventions escalate:
reflect (+ mask looping action) -> backtrack (restore snapshot) -> truncate
(shaped penalty). All firings are logged into RolloutRecord.monitor_flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from slm_rl.agents.base import ActionDecision
from slm_rl.config.schema import MonitorConfig
from slm_rl.games.base import StepResult

InterventionKind = Literal["reflect", "mask_action", "backtrack", "truncate"]


@dataclass
class Intervention:
    kind: InterventionKind
    reason: str
    penalty: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class DoomLoopMonitor:
    """One instance per episode."""

    def __init__(self, cfg: MonitorConfig):
        self.cfg = cfg

    def observe(
        self,
        decision: ActionDecision,
        result: StepResult,
        state_hash: str,
    ) -> Intervention | None:
        """Record the step; return an Intervention if one should fire."""
        raise NotImplementedError("Phase 1 (basic signals), Phase 3 (full suite)")

    def episode_stats(self) -> dict[str, float]:
        """Aggregates for metrics.json / the EvalGate: repeat_rate, revisits,
        interventions fired, invalid_rate."""
        raise NotImplementedError("Phase 1")
