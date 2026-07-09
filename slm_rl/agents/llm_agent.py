"""LLM player: enumerated legal-move menu prompting + the parsing ladder
(exact `ACTION:` line -> fuzzy fallback -> one retry with error feedback ->
random legal move substitution). See docs/DECISIONS.md D3."""

from __future__ import annotations

from slm_rl.agents.base import ActionDecision, Agent
from slm_rl.games.base import Observation
from slm_rl.inference.base import InferenceBackend


class LLMAgent(Agent):
    def __init__(self, backend: InferenceBackend, system_prompt: str, max_context_tokens: int = 2048):
        raise NotImplementedError("Phase 1")

    def act(self, obs: Observation, history: list[ActionDecision]) -> ActionDecision:
        raise NotImplementedError("Phase 1")
