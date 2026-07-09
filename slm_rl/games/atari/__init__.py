"""Atari via ALE — Phase 3 (Freeway first, RAM -> text observations).

Default execution is in-process `ale-py` (no Docker, 8GB-safe); the OpenEnv
Docker client mode lives in `slm_rl.bridges.openenv_bridge`. RAM decoding is
a per-game plugin file under `ram_maps/`.
"""

from slm_rl.games.base import Game, Observation, StepResult, ActionSpec
from slm_rl.games.registry import register_game


@register_game("atari_freeway")
class FreewayGame(Game):
    def reset(self, seed=None) -> Observation:
        raise NotImplementedError("Phase 3: ALE/Freeway adapter not built yet")

    def step(self, action: ActionSpec) -> StepResult:
        raise NotImplementedError("Phase 3: ALE/Freeway adapter not built yet")

    def state_hash(self) -> str:
        raise NotImplementedError("Phase 3: ALE/Freeway adapter not built yet")

    def system_prompt(self) -> str:
        raise NotImplementedError("Phase 3: ALE/Freeway adapter not built yet")

    @classmethod
    def eval_suite(cls):
        raise NotImplementedError("Phase 3: ALE/Freeway adapter not built yet")
