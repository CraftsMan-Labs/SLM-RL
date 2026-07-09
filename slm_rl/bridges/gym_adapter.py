"""Gymnasium adapter: wraps an external `gymnasium.Env` (ALE with
obs_type="ram") into our `Game` contract via a per-game ObservationRenderer
(RAM -> text). Requires the [atari] extra."""

from __future__ import annotations

from abc import ABC, abstractmethod

from slm_rl.games.base import ActionSpec, Game


class ObservationRenderer(ABC):
    """Per-game RAM/state -> text + legal actions. Freeway's implementation
    lives in slm_rl/games/atari/ram_maps/freeway.py."""

    @abstractmethod
    def render(self, raw_obs, info: dict) -> tuple[str, list[ActionSpec]]: ...


class GymnasiumGameAdapter(Game):
    def __init__(self, config, opponent=None, env_id: str = "", renderer: ObservationRenderer | None = None):
        raise NotImplementedError("Phase 3")
