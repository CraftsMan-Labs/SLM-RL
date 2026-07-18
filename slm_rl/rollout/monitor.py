"""DoomLoopMonitor: rollout-level anti-doom-loop machinery (D4).

Signals per step: identical-action streaks, action n-gram loops, state-hash
revisits, reward stagnation, invalid-action streaks. When a signal fires the
monitor escalates through the game's configured intervention ladder (e.g.
reflect -> truncate); every firing is logged into
RolloutRecord.monitor_flags by the runner.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

from slm_rl.agents.base import ActionDecision
from slm_rl.config.schema import MonitorConfig
from slm_rl.games.base import StepResult

InterventionKind = Literal["reflect", "mask_action", "truncate"]

_LADDER_ORDER: tuple[InterventionKind, ...] = (
    "reflect",
    "mask_action",
    "truncate",
)


@dataclass
class Intervention:
    kind: InterventionKind
    reason: str
    penalty: float = 0.0


class DoomLoopMonitor:
    """One instance per episode."""

    def __init__(self, cfg: MonitorConfig):
        self.cfg = cfg
        self._actions: list[str] = []
        self._state_visits: Counter[str] = Counter()
        self._invalid_streak = 0
        self._steps_since_reward = 0
        self.signal_counts: Counter[str] = Counter()
        self.interventions: list[Intervention] = []
        self._steps = 0

    def observe(
        self,
        decision: ActionDecision,
        result: StepResult,
        state_hash: str,
    ) -> Intervention | None:
        self._steps += 1
        self._actions.append(decision.action.id)
        self._state_visits[state_hash] += 1

        if decision.parse_status == "fallback_random":
            self._invalid_streak += 1
        else:
            self._invalid_streak = 0

        if result.reward > 1e-9:
            self._steps_since_reward = 0
        else:
            self._steps_since_reward += 1

        if result.terminated or result.truncated:
            return None  # episode over; nothing to intervene on

        fired = self._fired_signals(state_hash)
        if not fired:
            return None
        self.signal_counts.update(fired)

        kind = self._escalate()
        penalty = self.cfg.truncate_penalty if kind == "truncate" else 0.0
        intervention = Intervention(
            kind=kind,
            reason=", ".join(fired),
            penalty=penalty,
        )
        self.interventions.append(intervention)
        return intervention

    def episode_stats(self) -> dict[str, Any]:
        return {
            "steps": self._steps,
            "signals": dict(self.signal_counts),
            "interventions": len(self.interventions),
            "intervention_kinds": [iv.kind for iv in self.interventions],
            "max_state_revisits": max(self._state_visits.values(), default=0),
        }

    def _fired_signals(self, state_hash: str) -> list[str]:
        fired: list[str] = []
        if self._current_streak() >= self.cfg.action_repeat_threshold:
            fired.append("action_repeat")
        if self._has_ngram_loop():
            fired.append("ngram_loop")
        if self._state_visits[state_hash] >= self.cfg.state_revisit_threshold:
            fired.append("state_revisit")
        if self._steps_since_reward >= self.cfg.reward_stagnation_window:
            fired.append("reward_stagnation")
        if self._invalid_streak >= self.cfg.invalid_streak_threshold:
            fired.append("invalid_streak")
        return fired

    def _current_streak(self) -> int:
        streak = 0
        last = self._actions[-1]
        for action in reversed(self._actions):
            if action != last:
                break
            streak += 1
        return streak

    def _has_ngram_loop(self) -> bool:
        k = self.cfg.ngram_loop_threshold
        for n in range(2, self.cfg.ngram_max_n + 1):
            window = n * k
            if len(self._actions) < window:
                continue
            tail = self._actions[-window:]
            gram = tail[-n:]
            if all(tail[i : i + n] == gram for i in range(0, window, n)):
                return True
        return False

    def _escalate(self) -> InterventionKind:
        ladder = [k for k in _LADDER_ORDER if k in self.cfg.interventions]
        if not ladder:
            ladder = ["truncate"]
        idx = min(len(self.interventions), len(ladder) - 1)
        return ladder[idx]
