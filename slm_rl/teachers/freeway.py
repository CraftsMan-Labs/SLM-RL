"""Heuristic Freeway teacher (plan 016): hold-UP baseline.

The plan asks for a car-avoiding variant IF the per-lane car_x mapping
verifies ("pause when a car is within a probed collision window in the
current lane"), keeping whichever variant scores higher. It does not
verify (see ram_maps/freeway.py docstring: the naive
`lane = (player_y - 6) // 12` estimate, checked against ~290 observed
hit-like events over a 2048-decision episode, produced no consistent
collision-zone car_x value for any lane -- values were scattered across
the full 0-159 range). Per plan 016 rule 2 ("an unverifiable variable is
dropped from the observation, never guessed"), no car-avoidance logic
ships; this teacher is pure hold-UP.

Measured (plan 016 probe, raw ALE, seed 0, hold-UP for a full episode):
2048 decisions, score 21 (matches the plan's stated "~21 in the standard
2:16 episode" and clears the floor of >=15 with a wide margin) -- getting
knocked back by traffic slows individual crossings but the episode timer
is long enough that persistent forward pressure alone crosses ~21 times.
"""

from __future__ import annotations

from slm_rl.agents.base import ActionDecision, Agent
from slm_rl.agents.llm_agent import build_messages
from slm_rl.games.base import Observation


class FreewayCrosserAgent(Agent):
    """Always moves UP. Deterministic per construction (no RNG draw in the
    hot path) -- `seed` is accepted for interface parity with
    `make_teacher` but unused, same convention as PongTrackerAgent."""

    def __init__(self, system_prompt: str, seed: int | None = None):
        self.system_prompt = system_prompt

    def act(self, obs: Observation) -> ActionDecision:
        s = obs.metadata["state"]
        player_y = s["player_y"]

        by_id = {a.id: a for a in obs.legal_actions}
        action = by_id["UP"]
        rationale = f"My chicken is at y={player_y}. Moving up toward the goal."
        completion = f"{rationale}\nACTION: {action.id}"
        return ActionDecision(
            action=action,
            raw_completion=completion,
            prompt_messages=build_messages(self.system_prompt, obs),
        )
