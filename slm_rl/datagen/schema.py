"""The dataset product: one `RolloutRecord` per decision step, streamed to
JSONL during play and consolidated to parquet afterwards. This schema is
versioned — bump `schema_version` on breaking changes and keep readers
backwards-compatible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = 1


@dataclass
class RolloutRecord:
    # provenance
    run_id: str
    generation: int
    game: str
    episode_id: str
    step_idx: int
    seed: int
    model_id: str
    adapter_ref: str | None
    opponent_id: str | None
    # the decision
    prompt_messages: list[dict[str, Any]]  # chat format, persisted verbatim
    completion: str
    parsed_action: str
    legal_actions: list[str]
    parse_status: str  # ok | retry_ok | fallback_random
    # outcome
    reward: float
    shaped_reward: float
    cum_reward: float
    terminated: bool
    truncated: bool
    outcome: str | None  # win | loss | draw | score:<n> (terminal steps only)
    # anti-doom-loop bookkeeping
    state_hash: str
    monitor_flags: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "RolloutRecord":
        return cls(**json.loads(line))
