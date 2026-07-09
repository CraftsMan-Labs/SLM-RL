"""Core game contract: everything in SLM-RL is built against these types.

Games are pure Python, seed-deterministic, text-native, and have no ML
dependencies. Observations carry a rendered text state plus an explicit list
of legal actions — the agent picks from an enumerated menu (design decision
D3), which also mirrors the `observation.legal_actions` shape of the OpenEnv
Atari reference example.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from slm_rl.config.schema import GameConfig
    from slm_rl.eval.suites import EvalSuite


@dataclass(frozen=True)
class ActionSpec:
    """A single legal action, canonical and menu-renderable."""

    id: str  # canonical string, e.g. "place_col_3", "buy Silver"
    label: str  # human/menu text shown to the model
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    """Model-facing view of the game state."""

    text: str  # rendered state (kept small: 8GB context budget)
    legal_actions: list[ActionSpec]
    turn: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)


class OpponentPolicy(ABC):
    """A non-learning (or frozen) player the env advances internally (D2)."""

    name: ClassVar[str] = "opponent"

    @abstractmethod
    def act(self, obs: Observation) -> ActionSpec: ...


class Game(ABC):
    """The game plugin contract (D7).

    Implementations must be deterministic given `reset(seed=...)` and must
    keep reward computation inside `step` (verifiable, rule-based rewards).
    """

    name: ClassVar[str]

    def __init__(self, config: "GameConfig", opponent: OpponentPolicy | None = None):
        self.config = config
        self.opponent = opponent

    @abstractmethod
    def reset(self, seed: int | None = None) -> Observation: ...

    @abstractmethod
    def step(self, action: ActionSpec) -> StepResult: ...

    @abstractmethod
    def state_hash(self) -> str:
        """Stable hash of the current state, for revisit detection."""

    def snapshot(self) -> bytes:
        """State checkpoint for the backtrack intervention. Override if the
        default pickle of the whole instance is wrong or wasteful."""
        return pickle.dumps(self.__dict__)

    def restore(self, snap: bytes) -> None:
        self.__dict__.update(pickle.loads(snap))

    @abstractmethod
    def system_prompt(self) -> str:
        """Rules preamble prepended to every episode's chat."""

    @classmethod
    def heuristic_opponents(cls) -> dict[str, OpponentPolicy]:
        """Named scripted opponents (e.g. Big Money for Dominion)."""
        return {}

    @classmethod
    @abstractmethod
    def eval_suite(cls) -> "EvalSuite":
        """Fixed-seed benchmark suite used by the EvalGate."""
