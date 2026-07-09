"""Scripted players: baseline opponents and datagen smoke-test agents."""

from __future__ import annotations

import random

from slm_rl.agents.base import ActionDecision, Agent
from slm_rl.games.base import ActionSpec, Observation, OpponentPolicy


class RandomBot(OpponentPolicy):
    """Uniform random legal move — ELO anchor and smoke-test opponent."""

    name = "random"

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def act(self, obs: Observation) -> ActionSpec:
        return self._rng.choice(obs.legal_actions)


class RandomAgent(Agent):
    """Agent wrapper around random play, for model-free pipeline smoke tests
    (`slm-rl rollout --agent random`)."""

    def __init__(self, seed: int | None = None):
        self._bot = RandomBot(seed)

    def act(self, obs: Observation, history: list[ActionDecision]) -> ActionDecision:
        action = self._bot.act(obs)
        return ActionDecision(action=action, raw_completion=f"ACTION: {action.id}")
