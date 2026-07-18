"""Game registry: `@register_game` for in-repo games, setuptools entry points
(group ``slm_rl.games``) for third-party plugins — the Catan/driving-sim
onboarding path.
"""

from __future__ import annotations

from importlib.metadata import entry_points

from slm_rl.games.base import Game

_REGISTRY: dict[str, type[Game]] = {}
_ENTRY_POINTS_SCANNED = False


def register_game(name: str):
    """Class decorator: ``@register_game("mastermind")``."""

    def deco(cls: type[Game]) -> type[Game]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def _scan_entry_points() -> None:
    global _ENTRY_POINTS_SCANNED
    if _ENTRY_POINTS_SCANNED:
        return
    _ENTRY_POINTS_SCANNED = True
    for ep in entry_points(group="slm_rl.games"):
        if ep.name in _REGISTRY:
            continue
        try:
            cls = ep.load()
        except Exception:
            # A plugin (or an optional built-in like atari without ale-py
            # installed) must never break the platform.
            continue
        cls.name = ep.name
        _REGISTRY[ep.name] = cls


def get_game(name: str) -> type[Game]:
    _scan_entry_points()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown game {name!r}. Available: {sorted(_REGISTRY)}"
        ) from None


def available_games() -> list[str]:
    _scan_entry_points()
    return sorted(_REGISTRY)
