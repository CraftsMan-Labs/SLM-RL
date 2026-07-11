"""Heuristic Space Invaders teacher (plan 009): the first non-exact teacher.
Space Invaders has no exact solver, but an aim-and-fire heuristic over the
decoded RAM (slm_rl.games.atari.ram_maps.space_invaders) scores far above
random and its reasoning is mechanical to verbalize (plan-002 style
rationale), which is all a warm-start teacher needs.

Bombs are NOT dodged: enemies_y (bomb-relevant) was excluded from decode()
as unverified (see ram_maps/space_invaders.py docstring) -- the heuristic's
edge is aim quality, not survival. The DQN teacher over vector_obs() is the
principled upgrade (ROADMAP Phase 3).
"""

from __future__ import annotations

import random

from slm_rl.agents.base import ActionDecision, Agent
from slm_rl.agents.llm_agent import build_messages
from slm_rl.games.base import Observation

# Calibrated empirically (plan 009 Step 3 + coordinator revision, scratch
# scripts through the real EpisodeRunner; full tables in the executor report):
# - offset/tolerance sweep ({0,8,16,24} x {2,4,8}, greedy, seeds 0-19):
#   offset=8/tolerance=8 best, mean cum_reward 5.1667 vs random 1.1167 (4.63x).
# - epsilon sweep ({0.05, 0.1, 0.2} at offset=8/tolerance=8): 0.05 keeps
#   >= 2x random on every measured draw (2.51x/3.17x/2.81x across agent
#   seeds 0/1/2, seeds 0-19); 0.10 dipped below 2x on two fair draws (1.99x
#   over 20 episodes, 1.94x over 40); 0.20 failed the 3-agent-seed average
#   (1.88x). Largest epsilon that KEEPS >= 2x: 0.05.
BLOCK_AIM_OFFSET = 8
AIM_TOLERANCE = 8
# Exploration keeps warm-start episodes diverse: greedy play on this env is
# seed-invariant (a fixed action script yields byte-identical RAM regardless
# of episode seed, see tests/test_space_invaders.py), so without it
# warmstart_episodes=1000 would be 1000 copies of ONE episode and
# select_episodes' max_duplicate_action_sequences quota would collapse the
# dataset. Diversity comes from self._rng advancing across decisions AND
# episodes (one agent instance serves the whole warm start).
EXPLORE_EPS = 0.05


class HeuristicInvaderAgent(Agent):
    """Aim-and-fire teacher with epsilon exploration: moves the cannon under
    the invader block's near column and fires when roughly aligned; with
    probability EXPLORE_EPS it instead plays a uniformly random legal action
    (honestly verbalized as such). Determinism guarantee is per
    (agent seed, episode order) -- the RNG advances across episodes -- not
    per episode seed alone (CODING_GUIDELINE Sec 1.4: explicit seed, same
    construction -> byte-identical decisions)."""

    def __init__(self, system_prompt: str, seed: int | None = None):
        self.system_prompt = system_prompt
        self._rng = random.Random(seed)

    def act(self, obs: Observation, history: list[ActionDecision]) -> ActionDecision:
        s = obs.metadata["state"]
        px = s["player_x"]
        bx = s["enemies_x"]
        in_flight = s["missile_in_flight"]

        if self._rng.random() < EXPLORE_EPS:
            # exploratory move: rationale must never claim aim-based
            # reasoning for a random choice
            action = self._rng.choice(obs.legal_actions)
            rationale = (
                f"My cannon is at x={px}, the invader block at x={bx}. "
                f"Trying {action.id} to vary my approach."
            )
            completion = f"{rationale}\nACTION: {action.id}"
            return ActionDecision(
                action=action,
                raw_completion=completion,
                prompt_messages=build_messages(self.system_prompt, obs),
            )

        dx = bx + BLOCK_AIM_OFFSET - px
        by_id = {a.id: a for a in obs.legal_actions}

        if abs(dx) <= AIM_TOLERANCE:
            action_id = "FIRE" if not in_flight else "NOOP"
            rationale = (
                f"My cannon is at x={px} and the invader block is at x={bx} — "
                f"lined up. Missile {'in flight, waiting' if in_flight else 'ready: firing'}."
            )
        else:
            action_id = "RIGHTFIRE" if dx > 0 else "LEFTFIRE"
            rationale = (
                f"My cannon is at x={px} but the invader block is at x={bx}, "
                f"{abs(dx)} to the {'right' if dx > 0 else 'left'} — moving that way "
                "and firing as I go."
            )

        action = by_id[action_id]
        completion = f"{rationale}\nACTION: {action.id}"

        return ActionDecision(
            action=action,
            raw_completion=completion,
            prompt_messages=build_messages(self.system_prompt, obs),
        )
