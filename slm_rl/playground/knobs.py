"""Declarative knob schema for the playground UI.

Defaults are never hardcoded here — `current_defaults()` reads them from the
repo's own `configs/default.yaml` / `configs/games/<game>.yaml` at request
time (via `slm_rl.config.loader`), so future config changes flow into the UI
automatically (see plan 013 maintenance notes). This module is stdlib +
`slm_rl.config` only: no gymnasium/ale_py/numpy imports (CODING_GUIDELINE
8GB rule — this package must import with no optional extras installed).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

KnobTarget = Literal["game", "game.monitor", "game.extra", "run.train", "run.teacher"]
KnobType = Literal["int", "float", "enum"]


@dataclass(frozen=True)
class Knob:
    key: str
    label: str
    target: KnobTarget
    type: KnobType
    min: float | None = None
    max: float | None = None
    choices: tuple[str, ...] | None = None  # for type == "enum"


# Config-backed knobs. `teacher` is handled specially (see materialize_knobs
# in experiments.py): it is not a plain field on any config model, it picks
# between "heuristic" (default) and "dqn" (which also sets
# run.teacher.dqn_checkpoint).
KNOBS: list[Knob] = [
    Knob("max_turns", "Max turns", "game", "int", min=1, max=10_000),
    Knob("action_repeat", "Action repeat", "game.extra", "int", min=1, max=10),
    Knob("score_scale", "Score scale", "game.extra", "float", min=1.0, max=1000.0),
    Knob("life_loss_penalty", "Life loss penalty", "game.extra", "float", min=-10.0, max=0.0),
    Knob("noop_start_max", "No-op start max", "game.extra", "int", min=0, max=200),
    Knob("action_repeat_threshold", "Action repeat threshold", "game.monitor", "int", min=1, max=1000),
    Knob("ngram_loop_threshold", "N-gram loop threshold", "game.monitor", "int", min=1, max=1000),
    Knob("state_revisit_threshold", "State revisit threshold", "game.monitor", "int", min=1, max=1000),
    Knob("reward_stagnation_window", "Reward stagnation window", "game.monitor", "int", min=1, max=5000),
    Knob("selection_quantile", "Selection quantile", "run.train", "float", min=0.0, max=1.0),
    Knob("episodes_per_generation", "Episodes per generation", "run.train", "int", min=1, max=100_000),
    Knob("warmstart_episodes", "Warm-start episodes", "run.teacher", "int", min=0, max=1_000_000),
    Knob("teacher", "Teacher", "run.teacher", "enum", choices=("heuristic", "dqn")),
]

# Experiment-level fields: control the quick-experiment subprocess, not the
# materialized config.
AGENT_CHOICES: tuple[str, ...] = ("solver", "random")
DEFAULT_EPISODES = 30
MAX_EPISODES = 200
DEFAULT_SEED = 20000

# Default checkpoint materialized into run.teacher.dqn_checkpoint when the
# `teacher` knob is set to "dqn" and the attendee doesn't override it. Points
# at the checkpoint trained via `slm-rl train-dqn` (plan 012); a missing
# file is a loud error at rollout time (make_teacher), never silent
# fallback to the heuristic teacher.
DEFAULT_DQN_CHECKPOINT = "runs/teachers/dqn-space-invaders.pt"


def _get_path(data: dict[str, Any], target: KnobTarget, key: str) -> Any:
    if target == "game":
        return data.get(key)
    if target == "game.monitor":
        return data.get("monitor", {}).get(key)
    if target == "game.extra":
        return data.get("extra", {}).get(key)
    if target == "run.train":
        return data.get("train", {}).get(key)
    if target == "run.teacher":
        return data.get("teacher", {}).get(key)
    raise ValueError(f"unknown knob target: {target!r}")  # pragma: no cover


def current_defaults(game: str, config_dir: Path | None = None) -> dict[str, Any]:
    """Read every knob's current default straight from the repo configs
    (never hardcoded — see module docstring)."""
    from slm_rl.config.loader import CONFIG_DIR, load_yaml

    config_dir = config_dir or CONFIG_DIR
    run_data = load_yaml(config_dir / "default.yaml")
    game_path = config_dir / "games" / f"{game}.yaml"
    game_data = load_yaml(game_path) if game_path.exists() else {}

    defaults: dict[str, Any] = {}
    for knob in KNOBS:
        if knob.key == "teacher":
            dqn_checkpoint = _get_path(run_data, "run.teacher", "dqn_checkpoint")
            defaults["teacher"] = "dqn" if dqn_checkpoint else "heuristic"
            continue
        source = game_data if knob.target.startswith("game") else run_data
        defaults[knob.key] = _get_path(source, knob.target, knob.key)
    return defaults


def knobs_schema(game: str, config_dir: Path | None = None) -> list[dict[str, Any]]:
    """`/api/knobs` payload: every knob's static schema plus its live
    default."""
    defaults = current_defaults(game, config_dir=config_dir)
    schema = []
    for knob in KNOBS:
        entry: dict[str, Any] = {
            "key": knob.key,
            "label": knob.label,
            "target": knob.target,
            "type": knob.type,
            "default": defaults.get(knob.key),
        }
        if knob.min is not None:
            entry["min"] = knob.min
        if knob.max is not None:
            entry["max"] = knob.max
        if knob.choices is not None:
            entry["choices"] = list(knob.choices)
        schema.append(entry)
    return schema
