"""Space Invaders via ALE (plan 008) — the first external-env game landed
through `GymnasiumGameAdapter`. Score game: gate compares `mean_score`, not
`win_rate`. Trains via `reject_sft` only (GRPO stays Mastermind-only).
"""

from __future__ import annotations

from slm_rl.bridges.gym_adapter import GymnasiumGameAdapter, ObservationRenderer
from slm_rl.games.atari.ram_maps import space_invaders as ram_map
from slm_rl.games.base import ActionSpec
from slm_rl.games.registry import register_game

# Human-readable labels for the 6 ALE action meanings this game uses.
_ACTION_LABELS = {
    "NOOP": "stand still",
    "FIRE": "fire",
    "RIGHT": "move right",
    "LEFT": "move left",
    "RIGHTFIRE": "move right and fire",
    "LEFTFIRE": "move left and fire",
}


class SpaceInvadersRenderer(ObservationRenderer):
    """RAM -> compact text (<= 12 lines, 8GB budget) + the 6 ALE actions."""

    def render(self, raw_obs, info: dict) -> tuple[str, list[ActionSpec]]:
        decoded = ram_map.decode(raw_obs)
        lives = info.get("lives")
        score = int(info.get("score", 0))
        lines = [f"Score: {score}. Lives: {lives}."]
        lines.append(
            f"Your cannon is at x={decoded['player_x']} (0=far left, 255=far right)."
        )
        lines.append(
            f"{decoded['invaders_left']} invaders remain; the block's x position is {decoded['enemies_x']}."
        )
        missile_state = (
            f"in flight (y={decoded['missile_y']})"
            if decoded["missile_in_flight"]
            else "ready to fire"
        )
        lines.append(f"Your missile: {missile_state}.")
        lines.append("Move to dodge enemy bombs and line up shots.")
        text = "\n".join(lines)

        action_ids = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]
        legal_actions = [
            ActionSpec(id=aid, label=_ACTION_LABELS[aid]) for aid in action_ids
        ]
        return text, legal_actions


@register_game("space-invaders")
class SpaceInvadersGame(GymnasiumGameAdapter):
    def __init__(self, config, opponent=None):
        super().__init__(
            config,
            opponent,
            env_id="ALE/SpaceInvaders-v5",
            renderer=SpaceInvadersRenderer(),
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Space Invaders. Shoot the invaders for points "
            "before they reach the bottom of the screen. You lose a life "
            "when an enemy bomb hits your cannon, so move away from bombs "
            "falling toward your position. Repeating a movement action is "
            "fine (holding a direction, spamming fire) — the mistake is "
            "standing still under a falling bomb. Fire when your missile is "
            "ready and lined up with an invader; move left or right to "
            "dodge and to line up your next shot."
        )

    @classmethod
    def eval_suite(cls):
        from slm_rl.eval.suites import EvalSuite

        return EvalSuite(
            game="space-invaders",
            seeds=tuple(range(10_000, 10_100)),
            primary_metric="mean_score",
        )
