"""YAML loading with defaults + override merging.

Precedence (low -> high): configs/default.yaml -> configs/games/<game>.yaml
-> user-supplied YAML -> CLI/env overrides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from slm_rl.config.schema import GameConfig, RunConfig, TierConfig

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `base` (returns a new dict)."""
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: Path | str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_run_config(
    game: str | None = None,
    config_path: Path | str | None = None,
    overrides: dict[str, Any] | None = None,
    config_dir: Path | None = None,
) -> RunConfig:
    config_dir = config_dir or CONFIG_DIR
    merged: dict[str, Any] = load_yaml(config_dir / "default.yaml")
    if game:
        merged = deep_merge(merged, {"game": game})
    if config_path:
        merged = deep_merge(merged, load_yaml(config_path))
    if overrides:
        merged = deep_merge(merged, {k: v for k, v in overrides.items() if v is not None})
    return RunConfig(**merged)


def load_game_config(game: str, config_dir: Path | None = None) -> GameConfig:
    config_dir = config_dir or CONFIG_DIR
    path = config_dir / "games" / f"{game}.yaml"
    data = load_yaml(path) if path.exists() else {}
    return GameConfig(**deep_merge({"name": game}, data))


def load_tiers(config_dir: Path | None = None) -> list[TierConfig]:
    config_dir = config_dir or CONFIG_DIR
    data = load_yaml(config_dir / "hardware.yaml")
    return [TierConfig(**tier) for tier in data.get("tiers", [])]
