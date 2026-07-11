"""Mastermind: guess a hidden color code; feedback is exact/wrong-position
counts. Phase 1 game — solitaire, deterministic given seed, small renders.
"""

from __future__ import annotations

import hashlib
import itertools
import random

from slm_rl.games.base import ActionSpec, Game, Observation, StepResult
from slm_rl.games.registry import register_game

ALL_COLORS = "RGBYOP"
COLOR_NAMES = {
    "R": "Red", "G": "Green", "B": "Blue", "Y": "Yellow", "O": "Orange", "P": "Purple",
}


def score_guess(guess: str, secret: str) -> tuple[int, int]:
    """Mastermind feedback: (exact, partial). Pure and symmetric — also used
    by the GRPO consistency reward without an engine instance."""
    exact = sum(g == s for g, s in zip(guess, secret))
    common = sum(min(guess.count(c), secret.count(c)) for c in set(guess))
    return exact, common - exact


@register_game("mastermind")
class MastermindGame(Game):
    def __init__(self, config, opponent=None):
        super().__init__(config, opponent)
        extra = config.extra
        self.code_length: int = int(extra.get("code_length", 4))
        self.num_colors: int = int(extra.get("num_colors", 6))
        self.allow_duplicates: bool = bool(extra.get("allow_duplicates", True))
        self.colors = ALL_COLORS[: self.num_colors]

        combos = itertools.product(self.colors, repeat=self.code_length)
        if not self.allow_duplicates:
            combos = (c for c in combos if len(set(c)) == len(c))
        self._actions = tuple(
            ActionSpec(id="".join(c), label="".join(c)) for c in combos
        )

        self._secret: str = ""
        self._history: list[tuple[str, int, int]] = []  # (guess, exact, partial)
        self._best_exact = 0

    def reset(self, seed: int | None = None) -> Observation:
        rng = random.Random(seed)
        self._secret = "".join(rng.choice(self.colors) for _ in range(self.code_length))
        if not self.allow_duplicates:
            self._secret = "".join(rng.sample(self.colors, self.code_length))
        self._history = []
        self._best_exact = 0
        return self._observation()

    def step(self, action: ActionSpec) -> StepResult:
        guess = action.id
        if len(guess) != self.code_length or any(c not in self.colors for c in guess):
            raise ValueError(f"Illegal guess {guess!r} reached the engine")

        exact, partial = self._score(guess)
        self._history.append((guess, exact, partial))

        won = exact == self.code_length
        out_of_turns = len(self._history) >= self.config.max_turns

        reward = 1.0 if won else 0.0
        shaping = (
            self.config.shaping_weight * 0.1 * max(0, exact - self._best_exact)
        )
        self._best_exact = max(self._best_exact, exact)

        info: dict = {"exact": exact, "partial": partial, "shaping": shaping}
        if won:
            info["outcome"] = "win"
        elif out_of_turns:
            info["outcome"] = "loss"

        return StepResult(
            observation=self._observation(),
            reward=reward,
            terminated=won,
            truncated=out_of_turns and not won,
            info=info,
        )

    def state_hash(self) -> str:
        raw = self._secret + "|" + ";".join(g for g, _, _ in self._history)
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def system_prompt(self) -> str:
        color_list = ", ".join(f"{c}={COLOR_NAMES[c]}" for c in self.colors)
        return (
            "You are playing Mastermind. A secret code of "
            f"{self.code_length} colors is hidden (colors: {color_list}; "
            f"{'duplicates allowed' if self.allow_duplicates else 'no duplicates'}). "
            "Each turn you guess a code and receive feedback: how many pegs are "
            "the right color in the right position (exact), and how many are the "
            "right color in the wrong position (partial). Use the feedback to "
            "narrow down the code. Never repeat a guess — it gives no new "
            "information. Strategy: guess codes consistent with ALL feedback "
            "so far. State briefly what the feedback rules out, then choose "
            "a consistent code."
        )

    @classmethod
    def eval_suite(cls):
        from slm_rl.eval.suites import EvalSuite

        return EvalSuite(
            game="mastermind",
            seeds=tuple(range(10_000, 10_500)),
            primary_metric="win_rate",
        )

    def _score(self, guess: str) -> tuple[int, int]:
        return score_guess(guess, self._secret)

    def _observation(self) -> Observation:
        turns_left = self.config.max_turns - len(self._history)
        if self._history:
            lines = [
                f"Guess {i + 1}: {g} -> {e} exact, {p} partial"
                for i, (g, e, p) in enumerate(self._history)
            ]
            text = "Guesses so far:\n" + "\n".join(lines)
        else:
            text = "No guesses yet."
        text += f"\nTurns remaining: {turns_left}."
        return Observation(
            text=text,
            legal_actions=self._actions,
            turn=len(self._history),
            metadata={
                "action_format": (
                    f"a {self.code_length}-letter code using the letters "
                    f"{'/'.join(self.colors)}, e.g. ACTION: "
                    f"{self.colors[:self.code_length] if self.allow_duplicates else self.colors[:self.code_length]}"
                ),
                # structured (guess, exact, partial) history so teachers and
                # menu pruners never parse the rendered text (HYBRID_RL.md)
                "history": [list(t) for t in self._history],
            },
        )


@register_game("mastermind-easy")
class MastermindEasyGame(MastermindGame):
    """Curriculum entry point: same engine, 3x4 = 64-code space (plan 006).

    A subclass rather than a re-registration of MastermindGame because
    register_game sets cls.name — aliasing the same class object would
    corrupt the standard game's identity (both names would report
    name="mastermind-easy").
    """
