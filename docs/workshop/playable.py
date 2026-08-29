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
    action_ids: tuple[str, ...] = ()
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
        self.action_ids = self.action_labels
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
        self.action_ids: tuple[str, ...] = ()
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
            self.action_ids = ()
            self.action_labels = ()
            return
        self.action_ids = tuple(
            str(getattr(action, "id", i))
            for i, action in enumerate(self._obs.legal_actions or [])
        )
        self.action_labels = tuple(
            str(getattr(action, "label", None) or getattr(action, "id", i))
            for i, action in enumerate(self._obs.legal_actions or [])
        )

    def _state(self) -> dict[str, Any]:
        return {"rgb": self.render_rgb(), **self.metrics()}


class GamePanel:
    """Controller for clickable play. Display is optional."""

    def __init__(
        self,
        env: PlayableEnv,
        *,
        max_auto_repeat: int = MAX_AUTO_REPEAT,
    ) -> None:
        self.env = env
        self.max_auto_repeat = max(1, int(max_auto_repeat))
        self.history: list[int] = []
        self.ui: Any = None
        self.keyboard: Any = None
        self.env.reset()

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self.history = []
        return self.env.reset(seed=seed)

    def press(self, action_id: int) -> dict[str, Any]:
        state = self.env.step(action_id)
        self.history.append(
            _clip_action(action_id, len(self.env.action_labels))
        )
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
        if self.keyboard is not None:
            try:
                self.keyboard.close()
            except Exception:
                pass
        self.env.close()


def action_for_key(key: str, action_ids: tuple[str, ...]) -> int | None:
    """Map browser Arrow/WASD/Space keys to the best available game action."""
    normalized = tuple(
        str(action).upper().replace(" ", "") for action in action_ids
    )
    key_name = str(key or "").lower()
    targets = {
        "arrowleft": ("LEFT",),
        "a": ("LEFT",),
        "arrowright": ("RIGHT",),
        "d": ("RIGHT",),
        "arrowup": ("UP", "RIGHT+A", "FIRE"),
        "w": ("UP", "RIGHT+A", "FIRE"),
        "arrowdown": ("DOWN",),
        "s": ("DOWN",),
        " ": ("FIRE", "RIGHT+A"),
        "space": ("FIRE", "RIGHT+A"),
        "x": ("FIRE", "RIGHT+A"),
    }.get(key_name)
    if not targets:
        return None
    for target in targets:
        if target in normalized:
            return normalized.index(target)
    for target in targets:
        for i, action in enumerate(normalized):
            if target in action:
                return i
    return None


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
    """Render a compact game viewport with buttons and keyboard controls."""
    try:
        import io

        import ipywidgets as widgets
        import numpy as np
        from IPython.display import display
        from PIL import Image as PILImage
    except Exception as exc:  # noqa: BLE001
        print(
            f"clickable controls unavailable ({exc}); "
            "use panel.press(action_id)"
        )
        print(panel.metrics())
        return panel

    shown = display_fn or display
    frame = widgets.Image(
        format="png",
        layout=widgets.Layout(
            width="480px",
            max_width="100%",
            height="360px",
            object_fit="contain",
            border="2px solid #3a342d",
        ),
    )
    title = widgets.HTML(
        value=(
            f"<div style='font:700 15px system-ui;color:#f2e7d5'>"
            f"{panel.env.game_id.replace('-', ' ').title()}</div>"
            "<div style='font:12px system-ui;color:#a98d6b'>"
            "Arrow keys / WASD to move · Space or X = action</div>"
        )
    )
    focus_btn = widgets.Button(
        description="🎮 Click here to enable keyboard controls",
        button_style="success",
        tooltip="Keep this button focused while using Arrow keys or WASD",
        layout=widgets.Layout(width="360px", max_width="100%"),
    )
    status = widgets.HTML(
        layout=widgets.Layout(
            width="480px",
            max_width="100%",
            padding="8px 10px",
            border="1px solid #3a342d",
        )
    )
    auto_n = widgets.IntSlider(
        value=min(4, panel.max_auto_repeat),
        min=1,
        max=panel.max_auto_repeat,
        description="frames",
        continuous_update=False,
        layout=widgets.Layout(width="250px"),
    )

    def refresh() -> None:
        metrics = panel.metrics()
        status.value = (
            f"<b>{metrics.get('status')}</b> · "
            f"reward {float(metrics.get('reward') or 0):.2f} · "
            f"score/distance "
            f"{metrics.get('score', metrics.get('distance', 0))} · "
            f"steps {metrics.get('steps', 0)}"
        )
        rgb = panel.env.render_rgb()
        if rgb is None:
            return
        arr = np.asarray(rgb)
        if arr.ndim != 3:
            return
        source = PILImage.fromarray(arr.astype("uint8"))
        source.thumbnail((480, 360), PILImage.Resampling.NEAREST)
        canvas = PILImage.new("RGB", (480, 360), (10, 9, 8))
        offset = (
            (canvas.width - source.width) // 2,
            (canvas.height - source.height) // 2,
        )
        canvas.paste(
            source,
            offset,
        )
        buf = io.BytesIO()
        canvas.save(buf, format="PNG")
        frame.value = buf.getvalue()

    def on_action(idx: int) -> Callable[[Any], None]:
        def _clicked(_btn: Any) -> None:
            panel.press(idx)
            refresh()

        return _clicked

    buttons = []
    for i, (action_id, label) in enumerate(
        zip(panel.env.action_ids, panel.env.action_labels)
    ):
        btn = widgets.Button(
            description=str(label),
            tooltip=str(action_id),
            button_style=(
                "info"
                if "FIRE" not in action_id and "+A" not in action_id
                else "warning"
            ),
            layout=widgets.Layout(
                width="145px",
                height="38px",
                margin="3px",
            ),
        )
        btn.on_click(on_action(i))
        buttons.append(btn)
    reset_btn = widgets.Button(
        description="↻ Reset",
        button_style="danger",
        layout=widgets.Layout(width="110px"),
    )
    reset_btn.on_click(lambda _btn: (panel.reset(), refresh()))
    auto_btn = widgets.Button(
        description="▶ Repeat last",
        button_style="success",
        layout=widgets.Layout(width="130px"),
    )

    def _auto(_btn: Any) -> None:
        if not panel.env.action_labels:
            return
        idx = panel.history[-1] if panel.history else 0
        panel.auto_repeat(idx, auto_n.value)
        refresh()

    auto_btn.on_click(_auto)
    controls = widgets.Box(
        buttons,
        layout=widgets.Layout(
            width="480px",
            max_width="100%",
            display="flex",
            flex_flow="row wrap",
            justify_content="center",
        ),
    )
    utility = widgets.HBox(
        [reset_btn, auto_btn, auto_n],
        layout=widgets.Layout(
            width="480px",
            max_width="100%",
            flex_flow="row wrap",
        ),
    )
    game = widgets.VBox(
        [title, focus_btn, frame, status, controls, utility],
        layout=widgets.Layout(
            width="510px",
            max_width="100%",
            padding="14px",
            border="1px solid #60584e",
        ),
    )
    panel.ui = game

    try:
        from ipyevents import Event

        keyboard = Event(
            source=focus_btn,
            watched_events=["keydown"],
            prevent_default_action=True,
        )

        def _keydown(event: dict[str, Any]) -> None:
            idx = action_for_key(
                str(event.get("key") or ""),
                panel.env.action_ids,
            )
            if idx is None or panel.env.metrics().get("done"):
                return
            panel.press(idx)
            refresh()

        keyboard.on_dom_event(_keydown)
        panel.keyboard = keyboard
    except Exception:
        title.value += (
            "<div style='font:12px system-ui;color:#d89b55'>"
            "Keyboard capture unavailable; use the controls below.</div>"
        )

    shown(game)
    refresh()
    return panel
