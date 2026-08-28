"""CPU-safe checks for workshop playable-game adapters and the control panel."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKSHOP = ROOT / "docs" / "workshop"
sys.path.insert(0, str(WORKSHOP))

from playable import (  # noqa: E402
    MAX_AUTO_REPEAT,
    AtariPlayable,
    GamePanel,
    MarioPlayable,
    make_playable,
    show_game_panel,
)


class _FakeAction:
    def __init__(self, id: str, label: str) -> None:
        self.id = id
        self.label = label


class _FakeObs:
    def __init__(self, turn: int, score: float) -> None:
        self.text = f"turn {turn}"
        self.legal_actions = [_FakeAction("LEFT", "left"), _FakeAction("RIGHT", "right")]
        self.turn = turn
        self.metadata = {"score": score}


class _FakeResult:
    def __init__(self, obs, reward: float, done: bool) -> None:
        self.observation = obs
        self.reward = reward
        self.terminated = done
        self.truncated = False
        self.info = {}


class FakeAtariGame:
    name = "boxing"

    def __init__(self, *, terminal_after: int = 3) -> None:
        self.terminal_after = terminal_after
        self.n = 0
        self.closed = False
        self._env = SimpleNamespace(unwrapped=SimpleNamespace(ale=SimpleNamespace(getScreenRGB=self._rgb)))

    def _rgb(self):
        return np.zeros((8, 10, 3), dtype=np.uint8)

    def reset(self, seed=None):
        self.n = 0
        return _FakeObs(0, 0.0)

    def step(self, action):
        self.n += 1
        done = self.n >= self.terminal_after
        return _FakeResult(_FakeObs(self.n, float(self.n)), 0.5, done)


class FakeMarioEnv:
    def __init__(self, *, terminal_after: int = 4) -> None:
        self.terminal_after = terminal_after
        self.n = 0
        self.closed = False

    def reset(self, *args, **kwargs):
        self.n = 0
        return np.zeros((24, 32, 3), dtype=np.uint8)

    def step(self, action):
        self.n += 1
        obs = np.full((24, 32, 3), int(action) + 1, dtype=np.uint8)
        done = self.n >= self.terminal_after
        return obs, 1.0, done, {"x_pos": 10 * self.n}

    def close(self):
        self.closed = True


def test_atari_adapter_reset_step_terminal_and_close():
    game = FakeAtariGame(terminal_after=2)
    env = AtariPlayable(game, game_id="boxing")
    state = env.reset(seed=1)
    assert state["done"] is False
    assert env.action_labels == ("left", "right")
    assert env.render_rgb().shape == (8, 10, 3)
    env.step(1)
    last = env.step(99)
    assert last["done"] is True
    assert last["status"] == "terminal"
    assert last["steps"] == 2
    assert last["reward"] == 1.0
    env.close()
    assert game._env is None


def test_mario_adapter_tracks_distance_and_closes():
    raw = FakeMarioEnv(terminal_after=3)
    env = MarioPlayable(raw)
    env.reset()
    env.step(1)
    last = env.step(0)
    assert last["distance"] == 20
    assert last["score"] == 20
    assert last["done"] is False
    last = env.step(0)
    assert last["done"] is True
    env.close()
    assert raw.closed is True


def test_game_panel_press_and_bounded_auto_repeat():
    env = AtariPlayable(FakeAtariGame(terminal_after=20), game_id="boxing")
    panel = GamePanel(env, max_auto_repeat=5)
    panel.press(1)
    assert panel.history == [1]
    last = panel.auto_repeat(0, 99)
    assert last["steps"] == 6
    assert len(panel.history) == 6
    panel.auto_repeat("nope", 3)
    assert panel.metrics()["steps"] == 9
    panel.close()


def test_auto_repeat_stops_at_terminal():
    env = AtariPlayable(FakeAtariGame(terminal_after=2), game_id="freeway")
    panel = GamePanel(env, max_auto_repeat=MAX_AUTO_REPEAT)
    last = panel.auto_repeat(0, 20)
    assert last["done"] is True
    assert last["steps"] == 2
    panel.close()


def test_make_playable_rejects_unknown_and_keeps_atari_when_mario_fails(monkeypatch):
    env, err = make_playable("pong")
    assert env is None
    assert "unknown game" in err

    import mario_lab

    monkeypatch.setattr(mario_lab, "ensure_mario_packages", lambda install=False: (False, "import failed: test"))
    env, err = make_playable("mario", install_mario=False)
    assert env is None
    assert "import failed" in err


def test_show_game_panel_returns_panel(capsys):
    env = AtariPlayable(FakeAtariGame(), game_id="boxing")
    panel = GamePanel(env)
    shown = show_game_panel(panel, display_fn=lambda *_a, **_k: None)
    assert shown is panel
    panel.close()
