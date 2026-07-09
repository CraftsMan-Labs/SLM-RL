"""EpisodeRunner: drives Agent x Game with the DoomLoopMonitor wired in,
streaming every decision to the RolloutWriter."""

from __future__ import annotations

from slm_rl.agents.base import Agent
from slm_rl.config.schema import GameConfig
from slm_rl.datagen.writer import RolloutWriter
from slm_rl.games.base import Game


class EpisodeRunner:
    def __init__(self, game: Game, agent: Agent, game_cfg: GameConfig, writer: RolloutWriter | None = None):
        raise NotImplementedError("Phase 1")

    def run_episode(self, seed: int, episode_id: str) -> dict:
        """Plays one episode; returns episode summary (outcome, cum_reward,
        monitor stats)."""
        raise NotImplementedError("Phase 1")
