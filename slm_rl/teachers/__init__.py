"""Classical teachers around the LLM loop (HYBRID_RL.md, D11): solvers,
heuristics, and CleanRL-pattern DQN. Teachers implement the Agent ABC and
are barred from the eval gate.

Workshop keepers: boxing, space-invaders, freeway, demon-attack.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from slm_rl.config.schema import GameConfig


def _game(game_cfg: GameConfig):
    from slm_rl.games.registry import get_game

    return get_game(game_cfg.name)(game_cfg)


def _maybe_dqn(game_cfg, seed, dqn_checkpoint, heuristic_cls, heuristic_tag: str, *, heuristic_factory=None):
    """DQN if checkpoint set (loud miss), else heuristic/solver fallback."""
    from slm_rl.teachers.dqn_checkpoint import (
        find_dqn_checkpoint,
        is_legacy_space_invaders_default,
        missing_dqn_hint,
    )

    g = _game(game_cfg)

    def _fallback():
        if heuristic_factory is not None:
            return heuristic_factory(g, seed), heuristic_tag
        return heuristic_cls(g.system_prompt(), seed=seed), heuristic_tag

    if dqn_checkpoint is not None:
        path = dqn_checkpoint
        if not os.path.isfile(path):
            alt = find_dqn_checkpoint(game_cfg.name)
            if alt is not None:
                path = str(alt)
            elif is_legacy_space_invaders_default(dqn_checkpoint, game_cfg.name):
                print(
                    f"[teachers] {game_cfg.name}: ignoring legacy checkpoint "
                    f"{dqn_checkpoint!r} (no DQN for this game); using "
                    f"{heuristic_tag}",
                    flush=True,
                )
                return _fallback()
            else:
                raise ValueError(
                    f"dqn_checkpoint not found: {dqn_checkpoint!r}. "
                    + missing_dqn_hint(game_cfg.name)
                )
        from slm_rl.teachers.dqn import DQNTeacherAgent

        agent = DQNTeacherAgent(path, g.system_prompt(), seed=seed)
        return agent, agent.model_id
    return _fallback()


def _space_invaders(game_cfg: GameConfig, seed, dqn_checkpoint):
    from slm_rl.teachers.space_invaders_heuristic import HeuristicInvaderAgent

    return _maybe_dqn(
        game_cfg, seed, dqn_checkpoint, HeuristicInvaderAgent, "teacher:space_invaders_heuristic",
    )


def _freeway(game_cfg: GameConfig, seed, dqn_checkpoint):
    from slm_rl.teachers.freeway import FreewayCrosserAgent

    return _maybe_dqn(game_cfg, seed, dqn_checkpoint, FreewayCrosserAgent, "teacher:freeway_crosser")


def _boxing(game_cfg: GameConfig, seed, dqn_checkpoint):
    from slm_rl.teachers.boxing import BoxingPuncherAgent

    return _maybe_dqn(game_cfg, seed, dqn_checkpoint, BoxingPuncherAgent, "teacher:boxing_puncher")


def _demon_attack(game_cfg: GameConfig, seed, dqn_checkpoint):
    from slm_rl.teachers.demon_attack import DemonAttackTrackerAgent

    return _maybe_dqn(
        game_cfg, seed, dqn_checkpoint, DemonAttackTrackerAgent, "teacher:demon_attack_tracker",
    )


_TEACHERS: tuple[tuple[str, Callable], ...] = (
    ("space-invaders", _space_invaders),
    ("freeway", _freeway),
    ("boxing", _boxing),
    ("demon-attack", _demon_attack),
)


def _generic_dqn_teacher(game_cfg: GameConfig, seed, dqn_checkpoint: str):
    """Plugin games: DQN checkpoint + Game.vector_obs() → teacher (no heuristic)."""
    from slm_rl.teachers.dqn_checkpoint import find_dqn_checkpoint, missing_dqn_hint

    path = dqn_checkpoint
    if not os.path.isfile(path):
        alt = find_dqn_checkpoint(game_cfg.name)
        if alt is None:
            raise ValueError(
                f"dqn_checkpoint not found: {dqn_checkpoint!r}. "
                + missing_dqn_hint(game_cfg.name)
            )
        path = str(alt)

    g = _game(game_cfg)
    if not callable(getattr(g, "vector_obs", None)):
        raise ValueError(
            f"No teacher implemented for game {game_cfg.name!r} "
            f"(set dqn_checkpoint and implement vector_obs() for DQN warm-start)"
        )
    try:
        g.vector_obs()
    except NotImplementedError as exc:
        raise ValueError(
            f"No teacher implemented for game {game_cfg.name!r} "
            f"(vector_obs() is required for DQN warm-start)"
        ) from exc

    from slm_rl.teachers.dqn import DQNTeacherAgent

    agent = DQNTeacherAgent(path, g.system_prompt(), seed=seed)
    return agent, agent.model_id


def make_teacher(game_cfg: GameConfig, seed: int | None = None, dqn_checkpoint: str | None = None):
    """(Agent, model_id) for the game's teacher; ValueError if none exists.

    dqn_checkpoint: path to a trained DQN teacher checkpoint (slm-rl
    train-dqn). If set but unreadable, raises ValueError -- a silent
    fallback to the heuristic would poison a run the operator thought was
    DQN-taught.

    Plugin games with no Atari heuristic still get a teacher when
    ``dqn_checkpoint`` is set and the game implements ``vector_obs()``.
    """
    for prefix, builder in _TEACHERS:
        if game_cfg.name.startswith(prefix):
            return builder(game_cfg, seed, dqn_checkpoint)
    if dqn_checkpoint is not None:
        return _generic_dqn_teacher(game_cfg, seed, dqn_checkpoint)
    raise ValueError(f"No teacher implemented for game {game_cfg.name!r}")


def make_pruner(game_cfg: GameConfig, top_k: int = 10):
    """Menu pruner — none of the keeper games use one yet."""
    raise ValueError(f"No menu pruner implemented for game {game_cfg.name!r}")
