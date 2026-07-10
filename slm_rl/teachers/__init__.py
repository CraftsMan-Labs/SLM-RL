"""Classical teachers around the LLM loop (HYBRID_RL.md, D11): exact solvers
now, CleanRL-pattern DQN later. Teachers implement the Agent ABC and are
barred from the eval gate."""

from __future__ import annotations

from slm_rl.config.schema import GameConfig


def make_teacher(game_cfg: GameConfig, seed: int | None = None):
    """(Agent, model_id) for the game's teacher; ValueError if none exists."""
    if game_cfg.name == "mastermind":
        from slm_rl.games.registry import get_game
        from slm_rl.teachers.mastermind_solver import SolverAgent

        g = get_game("mastermind")(game_cfg)
        agent = SolverAgent(g.colors, g.code_length, g.allow_duplicates, g.system_prompt(), seed=seed)
        return agent, "teacher:mastermind_solver"
    raise ValueError(f"No teacher implemented for game {game_cfg.name!r}")


def make_pruner(game_cfg: GameConfig, top_k: int = 10):
    """The game's menu pruner; ValueError if none exists."""
    if game_cfg.name == "mastermind":
        from slm_rl.games.registry import get_game
        from slm_rl.teachers.pruner import ConsistentMenuPruner

        g = get_game("mastermind")(game_cfg)
        return ConsistentMenuPruner(g.colors, g.code_length, g.allow_duplicates, top_k=top_k)
    raise ValueError(f"No menu pruner implemented for game {game_cfg.name!r}")
