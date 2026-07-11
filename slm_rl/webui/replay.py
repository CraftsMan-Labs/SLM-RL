"""Live game screen: re-simulate an episode's recorded actions in a fresh
ALE env and yield real frames as PNGs.

Why this is safe to build (see plans/010-live-game-screen.md): plan 008
established that ALE with `repeat_action_probability=0.0` is byte-
deterministic given a seed and action script, and every rollout record
already carries `(seed, parsed_action)` per step. So instead of recording
frames during rollout (which would bloat records and touch the play path),
the viewer rebuilds the exact env the adapter used and replays the actions
that were already logged — a pure read-only observer (CODING_GUIDELINE
invariant 5), zero change to rollout/training.

8GB rule: gymnasium/ale_py (and the numpy they pull in) are imported lazily
inside `replay_frames`, never at module top level — core `slm_rl` (and this
module's mere presence) must still import with no optional extras
installed.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from slm_rl.config.loader import load_game_config
from slm_rl.webui.png import encode_rgb
from slm_rl.webui.tailer import iter_run_records

_POLL_INTERVAL = 0.5


class ReplayUnavailable(Exception):
    """Raised when an episode can't be re-simulated visually (non-Atari
    games, e.g. Mastermind — records carry everything needed for a future
    pure-JS board animation instead, but that's a separate feature)."""


def _peek_first_record(
    run_dir: Path, episode_id: str
) -> dict[str, Any] | None:
    """First available record for `episode_id`, or None if none exist yet.
    A one-shot catch-up scan (private stop event set immediately) so this
    never blocks — used only to learn `game`/`seed` before building the env."""
    stop = threading.Event()
    stop.set()
    for rec in iter_run_records(run_dir, stop=stop):
        if rec.get("episode_id") == episode_id:
            return rec
    return None


def replay_frames(
    run_dir: Path, episode_id: str, stop: threading.Event | None = None
) -> Iterator[bytes]:
    """Yield PNG-encoded frames for `episode_id` in `run_dir`, re-simulating
    the episode's recorded actions in a fresh env built from the game's
    config.

    Near-live follow: `iter_run_records` itself polls (0.5s) for new lines
    and new generation directories, so simply filtering its live stream to
    this `episode_id` and running until a terminal record is seen (or
    `stop` fires) gives us "watch a still-running episode" for free — no
    separate catch-up/poll phases, no re-scanning already-consumed records.
    """
    first = _peek_first_record(run_dir, episode_id)
    if first is None:
        return

    game = first.get("game")
    seed = first.get("seed")
    config = load_game_config(game)
    env_id = config.extra.get("env_id")
    if not env_id:
        raise ReplayUnavailable(f"no visual replay for this game: {game!r}")

    # Mirror slm_rl/bridges/gym_adapter.py's GymnasiumGameAdapter env
    # construction exactly (frozen copy read at commit b5de903 — that file
    # is out of scope here, being edited concurrently). Keep these two in
    # sync by hand; the determinism test is the guardrail.
    import ale_py
    import gymnasium as gym

    gym.register_envs(ale_py)
    env = gym.make(
        env_id,
        obs_type="ram",
        frameskip=4,
        repeat_action_probability=0.0,
        render_mode="rgb_array",
    )
    action_repeat = int(config.extra.get("action_repeat", 3))

    try:
        env.reset(seed=seed)
        action_ids = list(env.unwrapped.get_action_meanings())

        internal_stop = stop if stop is not None else threading.Event()
        for rec in iter_run_records(
            run_dir, poll_interval=_POLL_INTERVAL, stop=internal_stop
        ):
            if rec.get("episode_id") != episode_id:
                continue
            action = rec.get("parsed_action")
            if action not in action_ids:
                # Unknown action id (old/corrupt record) — stop replay
                # cleanly, don't raise: this is a read-only viewer.
                return
            action_id = action_ids.index(action)
            terminal = False
            for _ in range(action_repeat):
                _, _, terminated, truncated, _ = env.step(action_id)
                frame = env.render()
                yield encode_rgb(frame.tobytes(), frame.shape[1], frame.shape[0])
                if terminated or truncated:
                    terminal = True
                    break
            if terminal or rec.get("terminated") or rec.get("truncated"):
                return
    finally:
        env.close()
