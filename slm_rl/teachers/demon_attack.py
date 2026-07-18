"""Heuristic Demon Attack teacher (plan 026): track nearest demon x and fire.

Calibrated empirically (plan 026 probe, raw ALE, noop_start_max=30,
action_repeat=1, 8 seeds, 800-decision cap): mean score ~129 (min 90, max
175). action_repeat=2 scored higher on mean (~154) but with much wider
variance (min 30); ar=1 is the shipped choice for stabler demos. Wrap
handling (|dx|>128 flips direction) is required because player_x wraps
(see ram_maps/demon_attack.py).
"""

from __future__ import annotations

from slm_rl.agents.base import ActionDecision, Agent
from slm_rl.agents.llm_agent import build_messages
from slm_rl.games.base import Observation

DEAD_ZONE = 8


class DemonAttackTrackerAgent(Agent):
    """Tracks the nearest-in-x active demon and fires. Deterministic per
    construction -- `seed` kept for make_teacher parity."""

    def __init__(self, system_prompt: str, seed: int | None = None):
        self.system_prompt = system_prompt

    def act(self, obs: Observation) -> ActionDecision:
        s = obs.metadata["state"]
        px = s["player_x"]
        enemies = list(zip(s["enemy_x"], s["enemy_y"]))
        # ponytail: "active" = y in a mid-screen band; falls back to all three
        # if the band is empty (wave transitions).
        active = [e for e in enemies if 40 < e[1] < 160] or enemies
        target_x, target_y = min(active, key=lambda e: abs(e[0] - px))
        dx = target_x - px
        by_id = {a.id: a for a in obs.legal_actions}

        if abs(dx) > DEAD_ZONE:
            go_right = dx > 0
            if abs(dx) > 128:
                go_right = not go_right
            action_id = "RIGHTFIRE" if go_right else "LEFTFIRE"
            rationale = (
                f"Ship at x={px}, nearest demon at ({target_x},{target_y}) -- "
                f"moving {'right' if go_right else 'left'} and firing."
            )
        else:
            action_id = "FIRE"
            rationale = (
                f"Ship at x={px}, lined up with demon at ({target_x},{target_y}). Firing."
            )

        action = by_id[action_id]
        return ActionDecision(
            action=action,
            raw_completion=f"{rationale}\nACTION: {action.id}",
            prompt_messages=build_messages(self.system_prompt, obs),
        )
