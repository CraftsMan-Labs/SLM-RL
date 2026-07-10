"""Fixed-seed benchmark suites: every game ships one via
`Game.eval_suite()`; the EvalGate compares generations on it. Seeds are
fixed so evaluation is paired across generations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from slm_rl.agents.base import Agent
from slm_rl.config.schema import GameConfig
from slm_rl.games.base import Game


@dataclass(frozen=True)
class EvalSuite:
    game: str
    seeds: tuple[int, ...]                 # fixed; paired across generations
    primary_metric: str = "win_rate"       # or "mean_score"
    opponents: tuple[str, ...] = ()        # named opponents for competitive games
    metadata: dict = field(default_factory=dict)


def run_suite(
    suite: EvalSuite,
    make_agent: Callable[[], Agent],
    game_cls: type[Game],
    game_cfg: GameConfig,
    limit: int | None = None,
    pruner=None,
) -> dict:
    """Plays the suite (fresh game+agent per seed); returns gate-comparable
    metrics. `limit` caps episodes for smoke tests. `pruner` is for the
    side eval_pruned metric ONLY — the gate eval never passes it (teachers
    must not inflate measured skill)."""
    from slm_rl.rollout.runner import EpisodeRunner

    seeds = suite.seeds[:limit] if limit else suite.seeds
    wins = total_steps = invalid_steps = interventions = 0
    scores: list[float] = []

    for seed in seeds:
        runner = EpisodeRunner(game_cls(game_cfg), make_agent(), game_cfg, pruner=pruner)
        summary = runner.run_episode(seed, episode_id=f"eval-{seed}")
        wins += summary["outcome"] == "win"
        scores.append(summary["cum_reward"])
        total_steps += summary["steps"]
        invalid_steps += summary["invalid_steps"]
        interventions += summary["monitor"]["interventions"]

    n = len(seeds)
    metrics = {
        "episodes": n,
        "win_rate": wins / n if n else 0.0,
        "mean_score": sum(scores) / n if n else 0.0,
        "invalid_rate": invalid_steps / total_steps if total_steps else 0.0,
        "intervention_rate": interventions / n if n else 0.0,
        "mean_entropy": None,  # filled by backends that report logprobs
    }
    metrics["primary"] = metrics[suite.primary_metric]
    return metrics
