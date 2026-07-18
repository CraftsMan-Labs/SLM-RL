"""Heuristic Boxing teacher (plan 026): hold-UPFIRE baseline.

# ponytail: a chase-and-jab controller over player/enemy x,y was probed
(approach then FIRE when |dx|,|dy| below a threshold) and scored worse than
pure UPFIRE on ale-py 0.12 (chase means around -38 to -100 vs UPFIRE around
-25 across seeds 0-4 with noop_start). The CPU boxer is strong; Freeway-style
hold of one action is the shipped warm-start teacher. Diversity for SFT
comes from noop_start_max, not exploration.
"""

from __future__ import annotations

from slm_rl.agents.base import ActionDecision, Agent
from slm_rl.agents.llm_agent import build_messages
from slm_rl.games.base import Observation


class BoxingPuncherAgent(Agent):
    """Always UPFIRE. Deterministic per construction (no RNG in the hot
    path) -- `seed` is accepted for interface parity with `make_teacher` but
    unused, same convention as FreewayCrosserAgent."""

    def __init__(self, system_prompt: str, seed: int | None = None):
        self.system_prompt = system_prompt

    def act(self, obs: Observation) -> ActionDecision:
        s = obs.metadata["state"]
        by_id = {a.id: a for a in obs.legal_actions}
        action = by_id["UPFIRE"]
        rationale = (
            f"I am at ({s['player_x']},{s['player_y']}), opponent at "
            f"({s['enemy_x']},{s['enemy_y']}). Holding up+punch."
        )
        return ActionDecision(
            action=action,
            raw_completion=f"{rationale}\nACTION: {action.id}",
            prompt_messages=build_messages(self.system_prompt, obs),
        )
