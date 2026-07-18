"""Shared reward-shaping hook loader (plan 013 seam, plan 017 extraction).

`GymnasiumGameAdapter` (Atari games) supports an optional
`extra.reward_hook` config knob: a path to a Python file defining
`shape_reward(ctx: dict) -> float`. The mechanics — resolve the path
eagerly at construction (missing file is a loud `ValueError` now, never a
silent fallback), then load the module lazily on first use (so importing
the game module never pays for `importlib` machinery when no hook is set,
the common case) — live here once.

`tests/test_reward_hook.py` exercises this through `GymnasiumGameAdapter`
and must pass unmodified after this extraction (CODING_GUIDELINE: behavior-
preserving refactor). Stdlib-only (CODING_GUIDELINE 8GB rule): this module
must import with no optional extras installed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

ShapeReward = Callable[[dict[str, Any]], float]


def resolve_reward_hook_path(reward_hook: str | None) -> Path | None:
    """Eager existence check for `extra.reward_hook`. None -> None (no hook
    configured); a set-but-missing path is a `ValueError` at construction
    time, matching `dqn_checkpoint` in `make_teacher` (plan 012 doctrine:
    never a silent fallback)."""
    if reward_hook is None:
        return None
    path = Path(reward_hook)
    if not path.is_file():
        raise ValueError(f"reward_hook not found: {reward_hook!r}")
    return path


def load_shape_reward(path: Path) -> ShapeReward:
    """Import `path` as a standalone module and return its `shape_reward`
    callable. A module without one is a `ValueError` here (not at
    construction) -- mirrors the file-existence check, which IS eager."""
    spec = importlib.util.spec_from_file_location(
        "slm_rl_playground_reward_hook", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shape = getattr(module, "shape_reward", None)
    if not callable(shape):
        raise ValueError(
            f"reward_hook {str(path)!r} has no callable shape_reward(ctx)"
        )
    return shape


class RewardHook:
    """Small stateful wrapper: resolves the path eagerly (`__init__`), loads
    and caches the module lazily (`__call__` / `ensure_loaded`) on first use.
    Games that already do the eager-check-at-construction dance (holding
    `self._reward_hook_path` themselves) can use the two free functions
    above directly instead; this class is the convenience path for a game
    that wants a single field to hold."""

    def __init__(self, reward_hook: str | None):
        self.path: Path | None = resolve_reward_hook_path(reward_hook)
        self._shape: ShapeReward | None = None

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def ensure_loaded(self) -> ShapeReward:
        if self._shape is None:
            assert self.path is not None  # enabled implies a resolved path
            self._shape = load_shape_reward(self.path)
        return self._shape

    def __call__(self, ctx: dict[str, Any]) -> float:
        return float(self.ensure_loaded()(ctx))
