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

import random
import threading
from collections.abc import Iterator
from pathlib import Path

from slm_rl.config.loader import load_game_config
from slm_rl.webui.png import encode_rgb
from slm_rl.webui.tailer import iter_run_records

_POLL_INTERVAL = 0.5


def _apply_noop_start(env, *, seed: int, noop_start_max: int, action_ids: list[str]) -> None:
    """Mirror `GymnasiumGameAdapter.reset` no-op prefix (Mnih eval protocol).

    Without this, replay desyncs from logged actions whenever
    `noop_start_max > 0` — the visual episode can end early (e.g. invaders
    reach the ground) while the recorded teacher still had lives left.
    """
    if noop_start_max <= 0:
        return
    rng = random.Random(seed * 10_007 + 11)
    k = rng.randint(0, noop_start_max)
    noop_id = action_ids.index("NOOP") if "NOOP" in action_ids else 0
    for _ in range(k):
        _obs, _reward, terminated, truncated, _info = env.step(noop_id)
        if terminated or truncated:
            break


class ReplayUnavailable(Exception):
    """Raised when an episode can't be re-simulated visually (non-Atari
    games, e.g. Mastermind — records carry everything needed for a future
    pure-JS board animation instead, but that's a separate feature)."""


def replay_frames(
    run_dir: Path,
    episode_id: str,
    stop: threading.Event | None = None,
    *,
    generation: int | None = None,
    follow: bool = True,
) -> Iterator[bytes]:
    """Yield PNG-encoded frames for `episode_id` in `run_dir`, re-simulating
    the episode's recorded actions in a fresh env built from the game's
    config.

    Near-live follow: `iter_run_records` itself polls (0.5s) for new lines
    and new generation directories, so simply filtering its live stream to
    this `episode_id` and running until a terminal record is seen (or
    `stop` fires) gives us "watch a still-running episode" for free.

    `generation` (optional) scopes the JSONL walk to one gen dir and skips
    `json.loads` on lines that don't mention `episode_id` — required so
    Watch screen stays usable when sibling Atari rollouts are GB-scale.
    Pass `follow=False` for a single catch-up pass (no live poll).
    """
    # One pass (no separate peek scan): first matching record opens the env.
    import ale_py
    import gymnasium as gym

    gym.register_envs(ale_py)

    env = None
    action_ids: list[str] | None = None
    action_repeat = 3
    internal_stop = stop if stop is not None else threading.Event()

    try:
        for rec in iter_run_records(
            run_dir,
            poll_interval=_POLL_INTERVAL,
            stop=internal_stop,
            generation=generation,
            episode_id=episode_id,
            follow=follow,
        ):
            if env is None:
                game = rec.get("game")
                seed = rec.get("seed")
                config = load_game_config(game)
                env_id = config.extra.get("env_id")
                if not env_id:
                    raise ReplayUnavailable(f"no visual replay for this game: {game!r}")

                # Mirror slm_rl/bridges/gym_adapter.py's GymnasiumGameAdapter env
                # construction exactly. Keep these two in sync by hand; the
                # determinism test is the guardrail.
                env = gym.make(
                    env_id,
                    obs_type="ram",
                    frameskip=4,
                    repeat_action_probability=0.0,
                    render_mode="rgb_array",
                )
                action_repeat = int(config.extra.get("action_repeat", 3))
                env.reset(seed=seed)
                action_ids = list(env.unwrapped.get_action_meanings())
                _apply_noop_start(
                    env,
                    seed=int(seed),
                    noop_start_max=int(config.extra.get("noop_start_max", 0)),
                    action_ids=action_ids,
                )

            assert action_ids is not None
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
        if env is not None:
            env.close()
