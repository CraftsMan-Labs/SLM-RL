"""Dominion v1 — Phase 4 flagship (see docs/DECISIONS.md D8 for the scope:
2 players, 8 non-attack base cards, Big Money opponent, simplified turn).
Planned modules: cards.py, state.py, env.py, bots.py.
"""

from slm_rl.games.base import Game, Observation, StepResult, ActionSpec
from slm_rl.games.registry import register_game


@register_game("dominion")
class DominionGame(Game):
    def reset(self, seed=None) -> Observation:
        raise NotImplementedError("Phase 4: Dominion engine not built yet")

    def step(self, action: ActionSpec) -> StepResult:
        raise NotImplementedError("Phase 4: Dominion engine not built yet")

    def state_hash(self) -> str:
        raise NotImplementedError("Phase 4: Dominion engine not built yet")

    def system_prompt(self) -> str:
        raise NotImplementedError("Phase 4: Dominion engine not built yet")

    @classmethod
    def eval_suite(cls):
        raise NotImplementedError("Phase 4: Dominion engine not built yet")
