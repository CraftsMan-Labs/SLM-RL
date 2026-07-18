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
    Knob("max_turns", "Max turns", "game", "int", min=1, max=20_000),
    # Frozen-suite prefix for gate / baseline eval (plan 024). Workshop
    # default overrides to 1 so Evolve can finish in minutes, not hours.
    Knob("eval_episodes", "Eval episodes (gate)", "game", "int", min=1, max=300),
    Knob("action_repeat", "Action repeat", "game.extra", "int", min=1, max=10),
    Knob("score_scale", "Score scale", "game.extra", "float", min=1.0, max=1000.0),
    Knob("life_loss_penalty", "Life loss penalty", "game.extra", "float", min=-10.0, max=0.0),
    Knob("noop_start_max", "No-op start max", "game.extra", "int", min=0, max=200),
    # min=3 matches MonitorConfig defaults; max=5000 covers Atari games that
    # ship high thresholds (freeway 2000, boxing headroom) without lying vs
    # the number input's max attribute.
    Knob("action_repeat_threshold", "Action repeat threshold", "game.monitor", "int", min=3, max=5000),
    Knob("ngram_loop_threshold", "N-gram loop threshold", "game.monitor", "int", min=3, max=5000),
    Knob("state_revisit_threshold", "State revisit threshold", "game.monitor", "int", min=1, max=1000),
    Knob("reward_stagnation_window", "Reward stagnation window", "game.monitor", "int", min=1, max=5000),
    Knob("selection_quantile", "Selection quantile", "run.train", "float", min=0.0, max=1.0),
    # Tokens per action completion (rollout, eval, and GRPO generation all use
    # it). An action is "ACTION: 3" — the workshop default drops the 256 repo
    # value so each generation runs in minutes on CPU.
    Knob("max_completion_tokens", "Max tokens per action", "run.train", "int", min=4, max=256),
    # Workshop cap: 200 LLM×Atari episodes is ~20h on Docker CPU; UI max
    # steers attendees away from the repo train default (200).
    Knob("episodes_per_generation", "Episodes per generation", "run.train", "int", min=1, max=200),
    # GRPO wall-clock knobs (Docker CPU demos need tight caps; CLI keeps yaml).
    Knob("group_size", "GRPO group size", "run.train", "int", min=2, max=16),
    Knob("grpo_max_steps", "GRPO max steps", "run.train", "int", min=1, max=10_000),
    Knob("grpo_max_prompts", "GRPO max prompts", "run.train", "int", min=1, max=512),
    Knob("replay_generations", "Replay generations", "run.train", "int", min=1, max=20),
    Knob("warmstart_episodes", "Warm-start episodes", "run.teacher", "int", min=0, max=1_000_000),
    Knob("teacher", "Teacher", "run.teacher", "enum", choices=("heuristic", "dqn")),
]

# Playground create-form defaults that differ from configs/*.yaml.
# CLI still uses the repo yaml (200 rollouts, 100-gate). Playground keeps a
# workshop-speed collect + gate + GRPO budget so Docker CPU can finish a
# generation in ~20 minutes.
PLAYGROUND_DEFAULT_OVERRIDES: dict[str, Any] = {
    "episodes_per_generation": 2,
    "eval_episodes": 2,
    "max_turns": 30,
    # A game action is "ACTION: 3" — a handful of tokens. 256 is ~10x waste per
    # turn; 24 is enough for legal ACTION lines without truncating training.
    "max_completion_tokens": 24,
    "group_size": 2,
    "grpo_max_steps": 24,
    "grpo_max_prompts": 32,
    "replay_generations": 1,
}

# Experiment-level fields: control the quick-experiment subprocess, not the
# materialized config.
AGENT_CHOICES: tuple[str, ...] = ("solver", "random")
DEFAULT_EPISODES = 30
MAX_EPISODES = 200
DEFAULT_SEED = 20000

# Legacy constant — Space Invaders only. Prefer
# `slm_rl.teachers.dqn_checkpoint.expected_dqn_checkpoint(game)` /
# `find_dqn_checkpoint(game)`. Kept so older imports/tests don't break.
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
    for key, value in PLAYGROUND_DEFAULT_OVERRIDES.items():
        # Apply even when YAML default is null (e.g. grpo_max_steps) so the
        # create form shows the workshop budget, not an empty field.
        if key in defaults:
            defaults[key] = value
    return defaults


def _knob_help(key: str, game: str, default: Any) -> dict[str, str] | None:
    """Tutorial card for `key`, with this game's pre-filled default called out.

    Copy lives in `tutorial_content.CARDS` (plan 023). Monitor / game knobs
    append the live default so hover text matches the value in the form
    after a game switch (e.g. demon-attack action_repeat_threshold → 300).
    """
    from slm_rl.playground.tutorial_content import CARDS

    card = CARDS.get(key)
    if card is None:
        return None
    body = card["body"]
    if default is not None:
        body = (
            f"{body} Recommended default for {game}: {default} "
            f"(pre-filled from this game's config — leave it unless you "
            f"re-measure competent play)."
        )
    return {"title": card["title"], "body": body}


def knobs_schema(game: str, config_dir: Path | None = None) -> list[dict[str, Any]]:
    """`/api/knobs` payload: knobs that apply to `game`, plus live defaults.

    Relevance = default is not None after reading this game's YAML (and
    run defaults). Absent `game.extra` / `game.monitor` keys drop out.
    Shared run knobs always resolve from `default.yaml` and stay visible.
    Each entry includes `help` {title, body} for the Vue ⓘ hover cards.
    """
    defaults = current_defaults(game, config_dir=config_dir)
    schema = []
    for knob in KNOBS:
        default = defaults.get(knob.key)
        # ponytail: YAML presence is the allowlist; declare tunables in
        # configs/games/<game>.yaml (or default.yaml for run.*).
        if default is None:
            continue
        entry: dict[str, Any] = {
            "key": knob.key,
            "label": knob.label,
            "type": knob.type,
            "target": knob.target,
            "default": default,
        }
        if knob.min is not None:
            entry["min"] = knob.min
        if knob.max is not None:
            entry["max"] = knob.max
        if knob.choices is not None:
            entry["choices"] = list(knob.choices)
        help_card = _knob_help(knob.key, game, default)
        if help_card is not None:
            entry["help"] = help_card
        schema.append(entry)
    return schema
