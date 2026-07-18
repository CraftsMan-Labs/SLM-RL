"""LLM player: enumerated legal-move menu prompting + the parsing ladder
(exact `ACTION:` line -> index -> fuzzy -> one retry with error feedback ->
random legal move substitution). See docs/DECISIONS.md D3.

Prompting is stateless per turn: the observation text carries the episode
history (games render it), which keeps the context small (8GB budget rule).
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable

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


def generate_with_retry(
    backend: InferenceBackend,
    messages: list[dict],
    params: GenParams,
    legal: list[ActionSpec] | tuple[ActionSpec, ...],
    rng: random.Random,
    *,
    make_retry: Callable[[str], list[dict]],
    record: Callable[[list[dict]], list[dict]] | None = None,
) -> ActionDecision:
    """Generate → parse → one retry → fallback_random. Shared by LLM + VL."""
    rec = record or (lambda m: m)
    out = backend.generate([messages], params)[0]
    action = parse_action(out.text, legal)
    if action is not None:
        return ActionDecision(action, out.text, rec(messages), "ok", out.logprob)

    retry_messages = messages + make_retry(out.text)
    out2 = backend.generate([retry_messages], params)[0]
    action = parse_action(out2.text, legal)
    recorded = rec(retry_messages)
    if action is not None:
        return ActionDecision(action, out2.text, recorded, "retry_ok", out2.logprob)

    fallback = rng.choice(list(legal))
    return ActionDecision(fallback, out2.text, recorded, "fallback_random")


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

    def act(self, obs: Observation) -> ActionDecision:
        messages = build_messages(self.system_prompt, obs)
        retry_text = (
            "That was not a valid move. "
            + action_instruction(obs)
            + " Reply with a single line: ACTION: <your move>"
        )
        return generate_with_retry(
            self.backend, messages, self.params, obs.legal_actions, self._rng,
            make_retry=lambda text: [
                {"role": "assistant", "content": text},
                {"role": "user", "content": retry_text},
            ],
        )


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
