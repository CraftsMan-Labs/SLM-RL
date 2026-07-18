"""Cheap Game stand-in for unit tests that must not pull ALE.

FakeBackend emits ACTION: 1 → 1-based menu index → first legal action → win.
"""

from __future__ import annotations

from slm_rl.eval.suites import EvalSuite
from slm_rl.games.base import ActionSpec, Game, Observation, StepResult

_MENU = ("UP", "DOWN", "LEFT", "RIGHT")


class TinyGame(Game):
    name = "tiny"

    def __init__(self, config, opponent=None):
        super().__init__(config, opponent)
        self._turn = 0

    def vector_obs(self) -> list[float]:
        # Fixed 4-d stub so plugin DQN warm-start tests need no ALE.
        return [float(self._turn), 0.0, 0.0, 1.0]

    def _obs(self) -> Observation:
        return Observation(
            text="pick",
            legal_actions=[ActionSpec(id=a, label=a) for a in _MENU],
            turn=self._turn,
            metadata={"vector_obs": self.vector_obs()},
        )

    def reset(self, seed=None):
        self._turn = 0
        return self._obs()

    def step(self, action: ActionSpec) -> StepResult:
        self._turn += 1
        win = action.id == "UP"  # ACTION: 1 → menu index 1 → UP
        capped = self._turn >= self.config.max_turns
        return StepResult(
            observation=self._obs(),
            reward=1.0 if win else 0.0,
            terminated=win,
            truncated=(not win) and capped,
            info={"outcome": "win" if win else ("loss" if capped else None)},
        )

    def state_hash(self) -> str:
        return f"t{self._turn}"

    def system_prompt(self) -> str:
        return "play"

    @classmethod
    def eval_suite(cls):
        # 300 seeds so eval_episodes=300 tests still exercise a full suite.
        return EvalSuite(
            game="tiny",
            seeds=tuple(range(10_000, 10_300)),
            primary_metric="win_rate",
        )
