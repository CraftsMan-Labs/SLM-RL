"""Clickable playable-game adapters for the Colab workshop.

Mario and Atari share one interface so the same control panel can drive
either title. This module stays free of torch. Mario packages are imported
only when a Mario env is requested.
"""

from __future__ import annotations

from typing import Any, Callable

MAX_AUTO_REPEAT = 32
PLAYABLE_ATARI = ("boxing", "space-invaders", "freeway", "demon-attack")
PLAYABLE_GAMES = (*PLAYABLE_ATARI, "mario")


class PlayableEnv:
    """Common reset/step/render surface for workshop human play."""

    game_id: str = ""
    action_labels: tuple[str, ...] = ()

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        raise NotImplementedError

    def step(self, action_id: int) -> dict[str, Any]:
        raise NotImplementedError

    def render_rgb(self) -> Any:
        raise NotImplementedError

    def metrics(self) -> dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


def _clip_action(action_id: Any, n: int) -> int:
    try:
        idx = int(action_id)
    except (TypeError, ValueError):
        idx = 0
    if n <= 0:
        return 0
    return max(0, min(n - 1, idx))


def _status(*, done: bool, error: str = "") -> str:
    if error:
        return f"error: {error}"
    return "terminal" if done else "playing"


class MarioPlayable(PlayableEnv):
    game_id = "mario"

    def __init__(self, env: Any) -> None:
        from mario_lab import ACTION_NAMES

        self.action_labels = tuple(ACTION_NAMES)
        self._env = env
        self._obs: Any = None
        self._reward = 0.0
        self._steps = 0
        self._done = False
        self._x_pos = 0
        self._closed = False

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        from mario_lab import _reset

        self._obs = _reset(self._env)
        self._reward = 0.0
        self._steps = 0
        self._done = False
        self._x_pos = 0
        return self._state()

    def step(self, action_id: int) -> dict[str, Any]:
        from mario_lab import _step

        if self._done or self._closed:
            return self._state()
        action = _clip_action(action_id, len(self.action_labels))
        obs, reward, done, info = _step(self._env, action)
        self._obs = obs
        self._reward += float(reward)
        self._steps += 1
        self._done = bool(done)
        self._x_pos = int((info or {}).get("x_pos") or 0)
        return self._state()

    def render_rgb(self) -> Any:
        if self._obs is None:
            return None
        import numpy as np

        return np.asarray(self._obs)

    def metrics(self) -> dict[str, Any]:
        return {
            "game": self.game_id,
            "reward": self._reward,
            "score": self._x_pos,
            "distance": self._x_pos,
            "steps": self._steps,
            "done": self._done,
            "status": _status(done=self._done),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._env.close()
        except Exception:
            pass

    def _state(self) -> dict[str, Any]:
        return {"rgb": self.render_rgb(), **self.metrics()}


class AtariPlayable(PlayableEnv):
    def __init__(self, game: Any, game_id: str = "") -> None:
        self._game = game
        self.game_id = game_id or str(getattr(game, "name", "") or "atari")
        self._obs: Any = None
        self._reward = 0.0
        self._steps = 0
        self._done = False
        self._score = 0.0
        self._closed = False
        self.action_labels: tuple[str, ...] = ()

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self._obs = self._game.reset(seed=seed)
        self._reward = 0.0
        self._steps = 0
        self._done = False
        self._sync_labels()
        self._score = float((getattr(self._obs, "metadata", None) or {}).get("score") or 0)
        return self._state()

    def step(self, action_id: int) -> dict[str, Any]:
        if self._done or self._closed or self._obs is None:
            return self._state()
        legal = list(self._obs.legal_actions or [])
        if not legal:
            self._done = True
            return self._state()
        chosen = legal[_clip_action(action_id, len(legal))]
        result = self._game.step(chosen)
        self._obs = result.observation
        self._reward += float(result.reward)
        self._steps += 1
        self._done = bool(result.terminated or result.truncated)
        self._sync_labels()
        self._score = float((getattr(self._obs, "metadata", None) or {}).get("score") or 0)
        return self._state()

    def render_rgb(self) -> Any:
        env = getattr(self._game, "_env", None) or getattr(self._game, "env", None)
        try:
            return None if env is None else env.unwrapped.ale.getScreenRGB()
        except Exception:
            return None

    def metrics(self) -> dict[str, Any]:
        return {
            "game": self.game_id,
            "reward": self._reward,
            "score": self._score,
            "distance": self._score,
            "steps": self._steps,
            "done": self._done,
            "status": _status(done=self._done),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        env = getattr(self._game, "_env", None) or getattr(self._game, "env", None)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if hasattr(self._game, "_env"):
            self._game._env = None

    def _sync_labels(self) -> None:
        if self._obs is None:
            self.action_labels = ()
            return
        self.action_labels = tuple(
            str(getattr(action, "label", None) or getattr(action, "id", i))
            for i, action in enumerate(self._obs.legal_actions or [])
        )

    def _state(self) -> dict[str, Any]:
        return {"rgb": self.render_rgb(), **self.metrics()}


class GamePanel:
    """Controller for clickable play. Display is optional."""

    def __init__(self, env: PlayableEnv, *, max_auto_repeat: int = MAX_AUTO_REPEAT) -> None:
        self.env = env
        self.max_auto_repeat = max(1, int(max_auto_repeat))
        self.history: list[int] = []
        self.env.reset()

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self.history = []
        return self.env.reset(seed=seed)

    def press(self, action_id: int) -> dict[str, Any]:
        state = self.env.step(action_id)
        self.history.append(_clip_action(action_id, len(self.env.action_labels)))
        return state

    def auto_repeat(self, action_id: int, n: int) -> dict[str, Any]:
        try:
            count = int(n)
        except (TypeError, ValueError):
            count = 1
        count = max(0, min(count, self.max_auto_repeat))
        last = self.env.metrics()
        last["rgb"] = self.env.render_rgb()
        for _ in range(count):
            if self.env.metrics().get("done"):
                break
            last = self.press(action_id)
        return last

    def metrics(self) -> dict[str, Any]:
        return self.env.metrics()

    def close(self) -> None:
        self.env.close()


def make_playable(
    game_id: str,
    *,
    existing_game: Any = None,
    install_mario: bool = False,
) -> tuple[PlayableEnv | None, str]:
    """Build a playable env. Never raises for optional Mario failures."""
    name = str(game_id or "").strip().lower()
    if name == "mario":
        from mario_lab import ensure_mario_packages, try_make_mario_env

        ok, reason = ensure_mario_packages(install=install_mario)
        if not ok:
            return None, reason or "mario env unavailable"
        env, err = try_make_mario_env()
        if env is None:
            return None, err or "mario env unavailable"
        return MarioPlayable(env), ""
    if name not in PLAYABLE_ATARI:
        return None, f"unknown game: {game_id!r}"
    game = existing_game
    if game is None or str(getattr(game, "name", "")) != name:
        try:
            from slm_rl.config.loader import load_game_config
            from slm_rl.games.registry import get_game

            cfg = load_game_config(name)
            game = get_game(name)(cfg)
        except Exception as exc:  # noqa: BLE001
            return None, f"atari env unavailable: {exc}"
    return AtariPlayable(game, game_id=name), ""


def show_game_panel(
    panel: GamePanel,
    *,
    display_fn: Callable[..., Any] | None = None,
) -> GamePanel:
    """Render clickable controls. Falls back to printed metrics if widgets fail."""
    try:
        import io

        import ipywidgets as widgets
        import numpy as np
        from IPython.display import Image, display
        from PIL import Image as PILImage
    except Exception as exc:  # noqa: BLE001
        print(f"clickable controls unavailable ({exc}); use panel.press(action_id)")
        print(panel.metrics())
        return panel

    shown = display_fn or display
    frame = widgets.Image(format="png")
    status = widgets.HTML()
    auto_n = widgets.IntSlider(
        value=min(4, panel.max_auto_repeat),
        min=1,
        max=panel.max_auto_repeat,
        description="repeat",
    )

    def refresh() -> None:
        metrics = panel.metrics()
        status.value = (
            f"<b>{metrics.get('status')}</b> · reward {float(metrics.get('reward') or 0):.2f} · "
            f"score/distance {metrics.get('score', metrics.get('distance', 0))} · "
            f"steps {metrics.get('steps', 0)}"
        )
        rgb = panel.env.render_rgb()
        if rgb is None:
            return
        arr = np.asarray(rgb)
        if arr.ndim != 3:
            return
        buf = io.BytesIO()
        PILImage.fromarray(arr.astype("uint8")).save(buf, format="PNG")
        frame.value = buf.getvalue()

    def on_action(idx: int) -> Callable[[Any], None]:
        def _clicked(_btn: Any) -> None:
            panel.press(idx)
            refresh()

        return _clicked

    buttons = []
    for i, label in enumerate(panel.env.action_labels):
        btn = widgets.Button(description=str(label))
        btn.on_click(on_action(i))
        buttons.append(btn)
    reset_btn = widgets.Button(description="Reset")
    reset_btn.on_click(lambda _btn: (panel.reset(), refresh()))
    auto_btn = widgets.Button(description="Auto-repeat")

    def _auto(_btn: Any) -> None:
        if not panel.env.action_labels:
            return
        idx = panel.history[-1] if panel.history else 0
        panel.auto_repeat(idx, auto_n.value)
        refresh()

    auto_btn.on_click(_auto)
    shown(widgets.VBox([frame, status, widgets.HBox(buttons), widgets.HBox([reset_btn, auto_btn, auto_n])]))
    refresh()
    return panel
