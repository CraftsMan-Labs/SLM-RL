"""OpponentPool: samples opponents per episode from heuristic bots, the
latest champion, and a league of frozen past generations (LoRA adapter
swaps — never a second resident model on 8GB tiers). See D2."""

from __future__ import annotations

from slm_rl.games.base import OpponentPolicy


class OpponentPool:
    def __init__(self, mix: dict[str, float]):
        raise NotImplementedError("Phase 2")

    def sample(self, seed: int) -> OpponentPolicy:
        raise NotImplementedError("Phase 2")
