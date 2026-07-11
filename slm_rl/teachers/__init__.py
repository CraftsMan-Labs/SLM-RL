"""Classical teachers around the LLM loop (HYBRID_RL.md, D11): exact solvers,
a hand-written heuristic, and a CleanRL-pattern DQN (plan 012). Teachers
implement the Agent ABC and are barred from the eval gate."""

from __future__ import annotations

import os

from slm_rl.config.schema import GameConfig


def make_teacher(game_cfg: GameConfig, seed: int | None = None, dqn_checkpoint: str | None = None):
    """(Agent, model_id) for the game's teacher; ValueError if none exists.

    dqn_checkpoint: path to a trained DQN teacher checkpoint (slm-rl
    train-dqn). If set but unreadable, raises ValueError -- a silent
    fallback to the heuristic would poison a run the operator thought was
    DQN-taught."""
    if game_cfg.name.startswith("mastermind"):
        from slm_rl.games.registry import get_game
        from slm_rl.teachers.mastermind_solver import SolverAgent

        g = get_game(game_cfg.name)(game_cfg)
        agent = SolverAgent(g.colors, g.code_length, g.allow_duplicates, g.system_prompt(), seed=seed)
        return agent, "teacher:mastermind_solver"
    if game_cfg.name.startswith("space-invaders"):
        # Lazy import: ale-py must not be imported unless this branch runs
        # (CODING_GUIDELINE 8GB rule) -- get_game("space-invaders") pulls in
        # the atari game module, which itself lazy-imports ale_py/gymnasium
        # only inside GymnasiumGameAdapter._ensure_env.
        from slm_rl.games.registry import get_game

        g = get_game(game_cfg.name)(game_cfg)
        if dqn_checkpoint is not None:
            if not os.path.isfile(dqn_checkpoint):
                raise ValueError(f"dqn_checkpoint not found: {dqn_checkpoint!r}")
            from slm_rl.teachers.dqn import DQNTeacherAgent

            agent = DQNTeacherAgent(dqn_checkpoint, g.system_prompt(), seed=seed)
            return agent, "teacher:space_invaders_dqn"
        from slm_rl.teachers.space_invaders_heuristic import HeuristicInvaderAgent

        agent = HeuristicInvaderAgent(g.system_prompt(), seed=seed)
        return agent, "teacher:space_invaders_heuristic"
    raise ValueError(f"No teacher implemented for game {game_cfg.name!r}")


def make_pruner(game_cfg: GameConfig, top_k: int = 10):
    """The game's menu pruner; ValueError if none exists."""
    if game_cfg.name.startswith("mastermind"):
        from slm_rl.games.registry import get_game
        from slm_rl.teachers.pruner import ConsistentMenuPruner

        g = get_game(game_cfg.name)(game_cfg)
        return ConsistentMenuPruner(g.colors, g.code_length, g.allow_duplicates, top_k=top_k)
    raise ValueError(f"No menu pruner implemented for game {game_cfg.name!r}")
