"""Boxing via ALE (plan 026 Phase B). Score game: gate compares `mean_score`,
not `win_rate`. Signed score (player punches minus opponent punches). Trains
via reject_sft warm-start then GRPO on transformers.
"""

from __future__ import annotations

from slm_rl.bridges.gym_adapter import GymnasiumGameAdapter, ObservationRenderer
from slm_rl.games.atari.ram_maps import boxing as ram_map
from slm_rl.games.base import ActionSpec
from slm_rl.games.registry import register_game

# ponytail: ALE Boxing exposes 18 actions including diagonals; the workshop
# menu keeps the 4 cardinal moves + fire combos (10 ids) so the LLM action
# list stays short. Diagonals remain reachable via the raw env if a future
# teacher needs them.
_ACTION_LABELS = {
    "NOOP": "stand still",
    "FIRE": "punch",
    "UP": "move up",
    "RIGHT": "move right",
    "LEFT": "move left",
    "DOWN": "move down",
    "UPFIRE": "move up and punch",
    "RIGHTFIRE": "move right and punch",
    "LEFTFIRE": "move left and punch",
    "DOWNFIRE": "move down and punch",
}


class BoxingRenderer(ObservationRenderer):
    """RAM -> compact text (<= 12 lines, 8GB budget) + cardinal+fire menu."""

    def decode(self, raw_obs) -> dict:
        """Duck-typed hook (gym_adapter._observation): decoded RAM variables
        for non-LLM consumers (the heuristic teacher). Reused by `render`
        below so RAM is decoded once per step, not twice."""
        return ram_map.decode(raw_obs)

    def render(self, raw_obs, info: dict) -> tuple[str, list[ActionSpec]]:
        decoded = self.decode(raw_obs)
        score = int(info.get("score", 0))
        px, py = decoded["player_x"], decoded["player_y"]
        ex, ey = decoded["enemy_x"], decoded["enemy_y"]
        dx, dy = ex - px, ey - py
        lines = [
            f"Your score: {score}. "
            f"(RAM punches you={decoded['player_score_ram']} "
            f"opponent={decoded['enemy_score_ram']}.)"
        ]
        lines.append(
            f"You are at x={px}, y={py} "
            "(x: 30=left edge, ~77=right; y: 3=top, larger=down)."
        )
        lines.append(f"Opponent is at x={ex}, y={ey}.")
        # Relative geometry so the SLM does not need to subtract coords, plus
        # edge warnings so it stops spamming UP/LEFT into the ropes.
        horiz = "to your right" if dx > 2 else "to your left" if dx < -2 else "same x"
        vert = "below you" if dy > 2 else "above you" if dy < -2 else "same y"
        lines.append(f"Opponent is {horiz} and {vert} (dx={dx}, dy={dy}).")
        edges: list[str] = []
        if py <= 5:
            edges.append("at the TOP rope — UP does nothing; move down/toward opponent")
        if py >= 80:
            edges.append("at the BOTTOM rope — DOWN does nothing; move up/toward opponent")
        if px <= 32:
            edges.append("at the LEFT rope — prefer RIGHT toward the opponent")
        if px >= 74:
            edges.append("at the RIGHT rope — prefer LEFT")
        if edges:
            lines.append("Ring edges: " + "; ".join(edges) + ".")
        close = abs(dx) <= 12 and abs(dy) <= 12
        if close:
            lines.append("You are in punching range — prefer FIRE or a move+FIRE combo.")
        else:
            lines.append("Move toward the opponent and punch when close.")
        text = "\n".join(lines)

        legal_actions = [
            ActionSpec(id=aid, label=label) for aid, label in _ACTION_LABELS.items()
        ]
        return text, legal_actions


@register_game("boxing")
class BoxingGame(GymnasiumGameAdapter):
    def __init__(self, config, opponent=None):
        super().__init__(
            config,
            opponent,
            env_id="ALE/Boxing-v5",
            renderer=BoxingRenderer(),
        )

    def system_prompt(self) -> str:
        return (
            "You are playing Boxing. You control a boxer on the left side of "
            "the ring; the opponent is the CPU boxer. Move with UP/DOWN/LEFT/"
            "RIGHT and punch with FIRE (or a move+FIRE combo). Score a point "
            "when your punch lands; the opponent scores when theirs lands. "
            "The match ends when the clock runs out or one side reaches 100. "
            "Stay close enough to land punches, and do not stand still while "
            "the opponent is lining you up."
        )

    @classmethod
    def eval_suite(cls):
        from slm_rl.eval.suites import EvalSuite

        return EvalSuite(
            game="boxing",
            seeds=tuple(range(10_000, 10_100)),
            primary_metric="mean_score",
        )
