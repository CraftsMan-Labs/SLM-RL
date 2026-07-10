"""LLM player: enumerated legal-move menu prompting + the parsing ladder
(exact `ACTION:` line -> index -> fuzzy -> one retry with error feedback ->
random legal move substitution). See docs/DECISIONS.md D3.

Prompting is stateless per turn: the observation text carries the episode
history (games render it), which keeps the context small (8GB budget rule).
"""

from __future__ import annotations

import random
import re

from slm_rl.agents.base import ActionDecision, Agent
from slm_rl.games.base import ActionSpec, Observation
from slm_rl.inference.base import GenParams, InferenceBackend

MENU_LIMIT = 30  # above this, show a format instruction instead of a menu

_ACTION_RE = re.compile(r"ACTION:\s*([^\n]+)", re.IGNORECASE)


def action_instruction(obs: Observation) -> str:
    legal = obs.legal_actions
    if len(legal) <= MENU_LIMIT:
        menu = "\n".join(f"{i + 1}) {a.label}" for i, a in enumerate(legal))
        return f"Legal moves:\n{menu}\nAnswer with the move's number or name."
    fmt = obs.metadata.get("action_format", "one of the legal moves")
    return f"Your move must be {fmt} ({len(legal)} legal moves)."


def build_messages(system_prompt: str, obs: Observation) -> list[dict]:
    """The exact chat messages an LLMAgent would see. Module-level so teacher
    agents can stamp LLM-identical prompts into their records (required for
    the reject_sft warm-start distillation to transfer)."""
    user = obs.text + "\n\n" + action_instruction(obs)
    if nudge := obs.metadata.get("nudge"):
        user += f"\n\nIMPORTANT: {nudge}"
    user += "\nThink briefly if needed, then end with one line: ACTION: <your move>"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


class LLMAgent(Agent):
    def __init__(
        self,
        backend: InferenceBackend,
        system_prompt: str,
        gen_params: GenParams | None = None,
        seed: int | None = None,
    ):
        self.backend = backend
        self.system_prompt = system_prompt
        self.params = gen_params or GenParams()
        self._rng = random.Random(seed)

    def act(self, obs: Observation, history: list[ActionDecision]) -> ActionDecision:
        messages = self._build_messages(obs)
        out = self.backend.generate([messages], self.params)[0]
        action = parse_action(out.text, obs.legal_actions)
        if action is not None:
            return ActionDecision(action, out.text, messages, "ok", out.logprob)

        retry_messages = messages + [
            {"role": "assistant", "content": out.text},
            {
                "role": "user",
                "content": (
                    "That was not a valid move. "
                    + self._action_instruction(obs)
                    + " Reply with a single line: ACTION: <your move>"
                ),
            },
        ]
        out2 = self.backend.generate([retry_messages], self.params)[0]
        action = parse_action(out2.text, obs.legal_actions)
        if action is not None:
            return ActionDecision(action, out2.text, retry_messages, "retry_ok", out2.logprob)

        fallback = self._rng.choice(list(obs.legal_actions))
        return ActionDecision(fallback, out2.text, retry_messages, "fallback_random")

    def _build_messages(self, obs: Observation) -> list[dict]:
        return build_messages(self.system_prompt, obs)

    def _action_instruction(self, obs: Observation) -> str:
        return action_instruction(obs)


def extract_action_token(text: str) -> str | None:
    """The move token from the last `ACTION: ...` line, or None."""
    matches = _ACTION_RE.findall(text)
    if not matches:
        return None
    words = matches[-1].strip().split()
    if not words:
        return None
    return words[0].strip(" .,`'\"*<>[]()") or None


def parse_action(
    text: str,
    legal: list[ActionSpec] | tuple[ActionSpec, ...],
    strict: bool = False,
) -> ActionSpec | None:
    """The parsing ladder, agent-independent so it can be golden-tested.

    strict=True (GRPO rewards) drops the last-resort "mentioned anywhere"
    step so garbage completions don't luckily parse."""
    by_id = {a.id.upper(): a for a in legal}

    token = extract_action_token(text)
    if token is not None:
        if token.isdigit():
            idx = int(token)
            if 1 <= idx <= len(legal):
                return legal[idx - 1]
        upper = token.upper()
        if upper in by_id:
            return by_id[upper]
        # fuzzy: unique id contained in the token (e.g. "**RGBY**" or "col_3.")
        hits = [a for key, a in by_id.items() if key in upper]
        if len(hits) == 1:
            return hits[0]

    if strict:
        return None

    # last resort: the legal id mentioned last anywhere in the completion
    upper_text = text.upper()
    best, best_pos = None, -1
    for key, a in by_id.items():
        pos = upper_text.rfind(key)
        if pos > best_pos:
            best, best_pos = a, pos
    return best
