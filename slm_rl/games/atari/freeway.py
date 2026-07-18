"""Freeway via ALE (plan 016): a simple, single-goal crossing game -- the
"Phase 3 opener" candidate in docs/PLUGIN_GUIDE.md, landed second in the
ALE game pack. Score game: gate compares `mean_score`, not `win_rate`
(score = crossing count, always non-negative). Trains via `reject_sft` only
(GRPO stays Mastermind-only).
"""

from __future__ import annotations

from slm_rl.bridges.gym_adapter import GymnasiumGameAdapter, ObservationRenderer
from slm_rl.games.atari.ram_maps import freeway as ram_map
from slm_rl.games.base import ActionSpec
from slm_rl.games.registry import register_game

_ACTION_LABELS = {
    "NOOP": "stand still",
    "UP": "move up (toward the goal)",
    "DOWN": "move down (retreat)",
}


class FreewayRenderer(ObservationRenderer):
    """RAM -> compact text (<= 12 lines, 8GB budget) + the 3 ALE actions."""

    def decode(self, raw_obs) -> dict:
        """Duck-typed hook (gym_adapter._observation): decoded RAM variables
        for non-LLM consumers (the heuristic teacher). Reused by `render`
        below so RAM is decoded once per step, not twice."""
        return ram_map.decode(raw_obs)

    def render(self, raw_obs, info: dict) -> tuple[str, list[ActionSpec]]:
        decoded = self.decode(raw_obs)
        score = int(info.get("score", 0))
        player_y = decoded["player_y"]
        lines = [f"Crossings so far: {score}."]
        lines.append(
            f"Your chicken is at y={player_y} (6=start/bottom, 126=goal/top)."
        )
        # Cars are all exposed as flavor text; decode() does not claim to
        # know which lane is "current" (see ram_maps/freeway.py docstring),
        # so the renderer shows the full row rather than guessing one.
        cars = decoded["car_x"]
        lines.append(
            "Traffic (car x positions, lane 0=bottom to lane 9=top): "
            + ", ".join(str(c) for c in cars)
        )
        lines.append("Move up to cross; dodge traffic if a car looks close.")
        text = "\n".join(lines)

        action_ids = ["NOOP", "UP", "DOWN"]
        legal_actions = [
            ActionSpec(id=aid, label=_ACTION_LABELS[aid]) for aid in action_ids
        ]
        return text, legal_actions


@register_game("freeway")
class FreewayGame(GymnasiumGameAdapter):
    def __init__(self, config, opponent=None):
        super().__init__(
            config,
            opponent,
            env_id="ALE/Freeway-v5",
            renderer=FreewayRenderer(),
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Freeway. Your chicken must cross a busy road "
            "from the bottom to the top, scoring a point each time it "
            "reaches the goal, then starting over from the bottom. Moving "
            "up repeatedly is fine -- the road is crossed by persistent "
            "forward progress, not by clever dodging alone. Getting hit by "
            "a car knocks you back down the road but never ends the "
            "episode, so keep moving up."
        )

    @classmethod
    def eval_suite(cls):
        from slm_rl.eval.suites import EvalSuite

        return EvalSuite(
            game="freeway",
            seeds=tuple(range(10_000, 10_100)),
            primary_metric="mean_score",
        )
