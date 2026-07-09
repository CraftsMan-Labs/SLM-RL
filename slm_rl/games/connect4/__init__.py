"""Connect-4 — Phase 2 game (competitive; opponent pool + ELO league)."""

from slm_rl.games.base import Game, Observation, StepResult, ActionSpec
from slm_rl.games.registry import register_game


@register_game("connect4")
class Connect4Game(Game):
    def reset(self, seed=None) -> Observation:
        raise NotImplementedError("Phase 2: Connect-4 engine not built yet")

    def step(self, action: ActionSpec) -> StepResult:
        raise NotImplementedError("Phase 2: Connect-4 engine not built yet")

    def state_hash(self) -> str:
        raise NotImplementedError("Phase 2: Connect-4 engine not built yet")

    def system_prompt(self) -> str:
        raise NotImplementedError("Phase 2: Connect-4 engine not built yet")

    @classmethod
    def eval_suite(cls):
        raise NotImplementedError("Phase 2: Connect-4 engine not built yet")
