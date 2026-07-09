"""Fixed-seed benchmark suites: every game ships one via
`Game.eval_suite()`; the EvalGate compares generations on it."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalSuite:
    game: str
    seeds: tuple[int, ...]                 # fixed; paired across generations
    primary_metric: str = "win_rate"       # or "mean_score"
    opponents: tuple[str, ...] = ()        # named opponents for competitive games
    metadata: dict = field(default_factory=dict)


def run_suite(suite: EvalSuite, agent, game_cls) -> dict:
    """Plays the whole suite; returns metrics (primary metric, invalid rate,
    intervention rate, mean entropy)."""
    raise NotImplementedError("Phase 1")
