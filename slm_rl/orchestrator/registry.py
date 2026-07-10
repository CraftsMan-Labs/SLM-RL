"""Model registry: a JSON file holding the champion pointer, promotion
history, and ELO ratings. Rollback = the pointer simply doesn't move."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ModelRegistry:
    def __init__(self, path: Path):
        self.path = path
        if path.exists():
            self._data: dict[str, Any] = json.loads(path.read_text())
        else:
            self._data = {"champion": 0, "history": [], "elo": {}}

    @property
    def champion(self) -> int:
        return self._data["champion"]

    def promote(self, generation: int, reason: str) -> None:
        self._data["champion"] = generation
        self._data["history"].append(
            {"generation": generation, "event": "promoted", "reason": reason}
        )
        self._save()

    def reject(self, generation: int, reason: str) -> None:
        self._data["history"].append(
            {"generation": generation, "event": "rejected", "reason": reason}
        )
        self._save()

    @property
    def consecutive_failures(self) -> int:
        n = 0
        for entry in reversed(self._data["history"]):
            if entry["event"] == "rejected":
                n += 1
            else:
                break
        return n

    @property
    def next_generation(self) -> int:
        """First unrun generation number (gen 0 = base model). Resumable."""
        seen = [e["generation"] for e in self._data["history"]]
        return max(seen, default=0) + 1

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))
