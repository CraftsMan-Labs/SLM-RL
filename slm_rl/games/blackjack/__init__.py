"""Blackjack — Phase 2 game (stochastic; paired-seed evaluation)."""

from slm_rl.games.base import Game, Observation, StepResult, ActionSpec
from slm_rl.games.registry import register_game


@register_game("blackjack")
class BlackjackGame(Game):
    def reset(self, seed=None) -> Observation:
        raise NotImplementedError("Phase 2: Blackjack engine not built yet")

    def step(self, action: ActionSpec) -> StepResult:
        raise NotImplementedError("Phase 2: Blackjack engine not built yet")

    def state_hash(self) -> str:
        raise NotImplementedError("Phase 2: Blackjack engine not built yet")

    def system_prompt(self) -> str:
        raise NotImplementedError("Phase 2: Blackjack engine not built yet")

    @classmethod
    def eval_suite(cls):
        raise NotImplementedError("Phase 2: Blackjack engine not built yet")
