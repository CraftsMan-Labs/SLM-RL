"""Agent contract: an Agent turns Observations into ActionDecisions via an
InferenceBackend. The full prompt and raw completion are persisted verbatim
into the dataset (RolloutRecord)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from slm_rl.games.base import ActionSpec, Observation

ParseStatus = Literal["ok", "retry_ok", "fallback_random"]


@dataclass
class ActionDecision:
    action: ActionSpec
    raw_completion: str
    prompt_messages: list[dict[str, Any]] = field(default_factory=list)
    parse_status: ParseStatus = "ok"
    logprob: float | None = None


class Agent(ABC):
    @abstractmethod
    def act(self, obs: Observation) -> ActionDecision: ...
