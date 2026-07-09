"""Mastermind — Phase 1 game. Engine lands with the Phase 1 build."""

from slm_rl.games.base import Game, Observation, StepResult, ActionSpec
from slm_rl.games.registry import register_game


@register_game("mastermind")
class MastermindGame(Game):
    def reset(self, seed=None) -> Observation:
        raise NotImplementedError("Phase 1: Mastermind engine not built yet")

    def step(self, action: ActionSpec) -> StepResult:
        raise NotImplementedError("Phase 1: Mastermind engine not built yet")

    def state_hash(self) -> str:
        raise NotImplementedError("Phase 1: Mastermind engine not built yet")

    def system_prompt(self) -> str:
        raise NotImplementedError("Phase 1: Mastermind engine not built yet")

    @classmethod
    def eval_suite(cls):
        raise NotImplementedError("Phase 1: Mastermind engine not built yet")
