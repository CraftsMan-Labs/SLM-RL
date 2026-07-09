"""ELO league for competitive games. Random + heuristic bots stay in the
league permanently as fixed-rating anchors (prevents pool inflation from
only beating old selves)."""

from __future__ import annotations


class EloLeague:
    def __init__(self, anchor_ratings: dict[str, float] | None = None):
        raise NotImplementedError("Phase 2")

    def record_result(self, player_a: str, player_b: str, outcome: float) -> None:
        raise NotImplementedError("Phase 2")

    def ratings(self) -> dict[str, float]:
        raise NotImplementedError("Phase 2")
