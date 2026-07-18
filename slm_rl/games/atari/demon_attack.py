"""Demon Attack via ALE (plan 026 Phase B). Score game: gate compares
`mean_score`, not `win_rate`. Trains via `reject_sft` only (GRPO stays
Mastermind-only).
"""

from __future__ import annotations

from slm_rl.bridges.gym_adapter import GymnasiumGameAdapter, ObservationRenderer
from slm_rl.games.atari.ram_maps import demon_attack as ram_map
from slm_rl.games.base import ActionSpec
from slm_rl.games.registry import register_game

_ACTION_LABELS = {
    "NOOP": "stand still",
    "FIRE": "fire",
    "RIGHT": "move right",
    "LEFT": "move left",
    "RIGHTFIRE": "move right and fire",
    "LEFTFIRE": "move left and fire",
}


class DemonAttackRenderer(ObservationRenderer):
    """RAM -> compact text (<= 12 lines, 8GB budget) + the 6 ALE actions."""

    def decode(self, raw_obs) -> dict:
        """Duck-typed hook (gym_adapter._observation): decoded RAM variables
        for non-LLM consumers (the heuristic teacher). Reused by `render`
        below so RAM is decoded once per step, not twice."""
        return ram_map.decode(raw_obs)

    def render(self, raw_obs, info: dict) -> tuple[str, list[ActionSpec]]:
        decoded = self.decode(raw_obs)
        score = int(info.get("score", 0))
        lives = info.get("lives")
        lines = [f"Score: {score}. Lives: {lives}. Wave/level: {decoded['level']}."]
        lines.append(
            f"Your ship is at x={decoded['player_x']} "
            "(wraps the screen; ~17=left edge, ~245=right)."
        )
        enemies = ", ".join(
            f"({x},{y})" for x, y in zip(decoded["enemy_x"], decoded["enemy_y"])
        )
        lines.append(f"Demons (x,y): {enemies}.")
        if decoded["missile_in_flight"]:
            lines.append(f"Your missile is in flight (y={decoded['missile_y']}).")
        else:
            lines.append("Your missile is ready -- fire when lined up under a demon.")
        text = "\n".join(lines)

        legal_actions = [
            ActionSpec(id=aid, label=label) for aid, label in _ACTION_LABELS.items()
        ]
        return text, legal_actions


@register_game("demon-attack")
class DemonAttackGame(GymnasiumGameAdapter):
    def __init__(self, config, opponent=None):
        super().__init__(
            config,
            opponent,
            env_id="ALE/DemonAttack-v5",
            renderer=DemonAttackRenderer(),
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Demon Attack. Move your ship left and right along "
            "the bottom and fire upward at the descending demons. You score "
            "when a shot hits a demon; you lose a life when a demon or its "
            "projectile hits you. The ship wraps around the screen edges. "
            "Track a demon's x, move under it, and fire -- holding a direction "
            "while a target is off to one side is fine."
        )

    @classmethod
    def eval_suite(cls):
        from slm_rl.eval.suites import EvalSuite

        return EvalSuite(
            game="demon-attack",
            seeds=tuple(range(10_000, 10_100)),
            primary_metric="mean_score",
        )
