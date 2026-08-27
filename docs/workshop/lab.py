"""Runtime helpers for the Colab workshop notebook.

Imported by cells generated from ``build_nb.py``. Keep this module free of
torch / slm_rl imports so tests and the notebook can load it before the
heavy stack is ready.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

WORKSHOP_GAMES = ("boxing", "space-invaders", "freeway", "demon-attack")
MODES = ("QUICK", "FULL")
PRECISIONS = ("q4", "fp16", "auto")
PARSE_STATUSES = ("ok", "retry_ok", "fallback_random")
TRAIN_STRATEGIES = ("reject_sft", "grpo", "both")
GATE_GUESSES = ("promote", "reject")

BACKEND_FOR_PRECISION = {
    "q4": "transformers-4bit",
    "fp16": "transformers",
    "auto": None,
}

# QUICK stays workshop-length. FULL is labeled as long-running in the notebook.
BOUNDS = {
    "QUICK": {
        "seed": (0, 99),
        "dqn_decisions": (100, 8_000),
        "generations": (1, 2),
        "episodes_per_generation": (1, 8),
        "temperature": (0.0, 1.5),
        "max_tokens": (8, 64),
        "theater_episodes": (1, 2),
        "eval_limit": (1, 8),
        "replay_every": (1, 8),
        "row_index": (0, 10_000),
        "action_index": (1, 18),
        "max_turns": (8, 64),
        "miss_reward": (-1.0, 0.0),
        "win_reward": (0.1, 2.0),
        "min_improvement": (0.0, 2.0),
    },
    "FULL": {
        "seed": (0, 99_999),
        "dqn_decisions": (1_000, 300_000),
        "generations": (1, 3),
        "episodes_per_generation": (8, 50),
        "temperature": (0.0, 1.5),
        "max_tokens": (8, 128),
        "theater_episodes": (1, 5),
        "eval_limit": (8, 50),
        "replay_every": (1, 8),
        "row_index": (0, 10_000),
        "action_index": (1, 18),
        "max_turns": (32, 2500),
        "miss_reward": (-1.0, 0.0),
        "win_reward": (0.1, 2.0),
        "min_improvement": (0.0, 2.0),
    },
}


def clamp_int(value: Any, lo: int, hi: int, name: str) -> int:
    """Coerce to int and clip to [lo, hi]."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = lo
    clipped = max(lo, min(hi, parsed))
    if clipped != parsed:
        print(f"clamped {name}: {parsed} → {clipped} (allowed {lo}–{hi})")
    return clipped


def clamp_float(value: Any, lo: float, hi: float, name: str) -> float:
    """Coerce to float and clip to [lo, hi]."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = lo
    clipped = max(lo, min(hi, parsed))
    if abs(clipped - parsed) > 1e-12:
        print(f"clamped {name}: {parsed} → {clipped} (allowed {lo}–{hi})")
    return clipped


def bound(mode: str, name: str) -> tuple[float, float]:
    table = BOUNDS.get(mode, BOUNDS["QUICK"])
    return table[name]


def sanitize_run_name(name: str) -> str:
    """Filesystem-safe run id. Empty / junk becomes ``colab``."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", (name or "").strip()).strip("-._")
    return cleaned[:40] or "colab"


def resolve_backend(precision: str) -> str | None:
    if precision not in BACKEND_FOR_PRECISION:
        print(f"unknown PRECISION={precision!r}; falling back to q4")
        return BACKEND_FOR_PRECISION["q4"]
    return BACKEND_FOR_PRECISION[precision]


def resolve_game(name: str) -> str:
    key = (name or "").strip().lower()
    if key in WORKSHOP_GAMES:
        return key
    print(f"unknown GAME={name!r}; falling back to boxing")
    return "boxing"


def resolve_mode(name: str) -> str:
    key = (name or "").strip().upper()
    if key in MODES:
        return key
    print(f"unknown MODE={name!r}; falling back to QUICK")
    return "QUICK"


def resolve_choice(value: str, allowed: Iterable[str], default: str) -> str:
    key = (value or "").strip()
    lookup = {item.lower(): item for item in allowed}
    if key.lower() in lookup:
        return lookup[key.lower()]
    print(f"unknown choice {value!r}; falling back to {default!r}")
    return default


def pick_action(legal: list[Any], raw: str) -> Any:
    """Resolve a typed id, label, or 1-based index against ``legal`` actions."""
    if not legal:
        raise ValueError("no legal actions")
    text = (raw or "").strip()
    if not text:
        return legal[0]
    for action in legal:
        if str(getattr(action, "id", "")) == text:
            return action
        if str(getattr(action, "label", "")).lower() == text.lower():
            return action
    if text.isdigit():
        idx = int(text)
        if 1 <= idx <= len(legal):
            return legal[idx - 1]
    print(f"action {raw!r} not in the menu — using {legal[0].id!r}")
    return legal[0]


def grade(prediction: str, actual: str) -> str:
    """One-line prediction readout. Case-insensitive."""
    pred = (prediction or "").strip().lower()
    act = (actual or "").strip().lower()
    if pred in {"", "not sure", "skip"}:
        return f"result: {actual}"
    if pred == act:
        return f"you predicted {prediction} — correct."
    return f"you predicted {prediction}; the result was {actual}."


def scorecard(title: str, rows: list[tuple[str, Any]]) -> None:
    """Compact two-column result table."""
    print(f"=== {title} ===")
    if not rows:
        print("  (empty)")
        return
    width = max(len(str(key)) for key, _ in rows)
    for key, value in rows:
        print(f"  {str(key):<{width}}  {value}")


def require_names(namespace: dict[str, Any], *names: str) -> None:
    missing = [name for name in names if name not in namespace or namespace[name] is None]
    if missing:
        raise RuntimeError(
            "Run the earlier cells first. Missing: " + ", ".join(missing)
        )
