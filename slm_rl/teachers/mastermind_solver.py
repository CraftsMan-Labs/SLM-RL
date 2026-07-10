"""Exact Mastermind teacher: consistency filtering is a complete solver for
this game, so no DQN training is needed (HYBRID_RL.md build-order step 1).

Random-consistent play wins standard Mastermind (4 pegs / 6 colors) well
within 12 turns; the solver never sees the secret — it deduces from the same
feedback the LLM sees.
"""

from __future__ import annotations

import itertools
import random

from slm_rl.agents.base import ActionDecision, Agent
from slm_rl.agents.llm_agent import build_messages
from slm_rl.games.base import Observation
from slm_rl.games.mastermind.env import score_guess


def consistent_candidates(
    colors: str,
    code_length: int,
    dup_ok: bool,
    history: list,
) -> list[str]:
    """All codes consistent with every (guess, exact, partial) in history.

    Never empty (the secret is always consistent), and excludes every played
    non-winning guess by construction: score_guess(g, g) == (code_length, 0),
    which can't match g's own feedback unless g won.
    """
    combos = itertools.product(colors, repeat=code_length)
    if not dup_ok:
        combos = (c for c in combos if len(set(c)) == len(c))
    codes = ("".join(c) for c in combos)
    return [
        code
        for code in codes
        if all(score_guess(g, code) == (e, p) for g, e, p in history)
    ]


class SolverAgent(Agent):
    """Teacher-as-Agent: plays a seeded-random consistent candidate. Builds
    LLM-identical prompt_messages so its records feed reject_sft unchanged.
    """

    # ponytail: random-consistent; Knuth minimax if avg turns ever matters

    def __init__(self, colors: str, code_length: int, dup_ok: bool, system_prompt: str, seed: int | None = None):
        self.colors = colors
        self.code_length = code_length
        self.dup_ok = dup_ok
        self.system_prompt = system_prompt
        self._rng = random.Random(seed)

    def act(self, obs: Observation, history: list[ActionDecision]) -> ActionDecision:
        cands = consistent_candidates(
            self.colors, self.code_length, self.dup_ok, obs.metadata.get("history", [])
        )
        by_id = {a.id: a for a in obs.legal_actions}
        # intersect with the (possibly pruned) menu so the choice is always
        # menu-legal; fall back to the menu if the intersection is empty
        pool = [c for c in cands if c in by_id] or list(by_id)
        guess = self._rng.choice(pool)
        return ActionDecision(
            action=by_id[guess],
            raw_completion=f"ACTION: {guess}",
            prompt_messages=build_messages(self.system_prompt, obs),
        )
