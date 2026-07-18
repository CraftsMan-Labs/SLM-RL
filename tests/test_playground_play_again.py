"""Plan 026 Phase G: optional play-again from scoreboard.

Route validation + launch smoke with mocked subprocess — never loads a
model (CODING_GUIDELINE). Reuses the theater single-flight lock.
"""

from __future__ import annotations

import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from slm_rl.playground import experiments as exp_mod
from slm_rl.playground import server as pg_server_mod
from slm_rl.playground.experiments import (
    Busy,
    InvalidExperiment,
    create_experiment,
    launch_play_again,
    launch_theater,
    resolve_play_again_generation,
)
from slm_rl.playground.page import PAGE


class _FakePopen:
    pid = 1

    def poll(self):
        return None


class _ServerContext:
    def __init__(self, home: Path, game: str = "space-invaders"):
        handler_cls = pg_server_mod._make_handler(home, game)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_ServerContext":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5.0)

    def post(self, path: str, payload: dict) -> http.client.HTTPResponse:
        body = json.dumps(payload).encode("utf-8")
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        return conn.getresponse()

    def get(self, path: str) -> http.client.HTTPResponse:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", path)
        return conn.getresponse()


@pytest.fixture(autouse=True)
def _reset_locks():
    for kind in ("quick", "evolve", "theater", "bake"):
        exp_mod._ACTIVE[kind] = None
        exp_mod._ACTIVE_OWNER[kind] = None
    yield
    for kind in ("quick", "evolve", "theater", "bake"):
        exp_mod._ACTIVE[kind] = None
        exp_mod._ACTIVE_OWNER[kind] = None


def _seed_run_config(exp, *, champion: int = 0) -> None:
    """Minimal evolve artifacts so play-again / theater resolvers can run."""
    exp.run_dir.mkdir(parents=True, exist_ok=True)
    (exp.run_dir / "run_config.yaml").write_text(
        "run_id: "
        + exp.run_id
        + "\nhome: "
        + str(exp.path)
        + "\ngame: space-invaders\ntier: any-8gb\n",
        encoding="utf-8",
    )
    (exp.run_dir / "registry.json").write_text(
        json.dumps({"champion": champion, "history": []}),
        encoding="utf-8",
    )
    if champion > 0:
        adapter = exp.run_dir / "generations" / f"gen_{champion:03d}" / "adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")


# --- resolve_play_again_generation ----------------------------------------


def test_resolve_gen_explicit(tmp_path: Path):
    exp = create_experiment(tmp_path, "space-invaders", "pa-gen", knob_values={})
    assert resolve_play_again_generation(exp, gen=2, champion=False) == 2


def test_resolve_champion_reads_registry(tmp_path: Path):
    exp = create_experiment(tmp_path, "space-invaders", "pa-champ", knob_values={})
    _seed_run_config(exp, champion=3)
    assert resolve_play_again_generation(exp, gen=None, champion=True) == 3


def test_resolve_champion_missing_rejects(tmp_path: Path):
    exp = create_experiment(tmp_path, "space-invaders", "pa-nochamp", knob_values={})
    _seed_run_config(exp, champion=0)
    with pytest.raises(InvalidExperiment, match="no promoted champion"):
        resolve_play_again_generation(exp, gen=None, champion=True)


def test_resolve_requires_gen_or_champion(tmp_path: Path):
    exp = create_experiment(tmp_path, "space-invaders", "pa-need", knob_values={})
    with pytest.raises(InvalidExperiment, match="provide gen"):
        resolve_play_again_generation(exp, gen=None, champion=False)


# --- launch smoke (mock subprocess) ---------------------------------------


def test_launch_play_again_cmd_shape(tmp_path: Path, monkeypatch):
    exp = create_experiment(tmp_path, "space-invaders", "pa-cmd", knob_values={})
    _seed_run_config(exp, champion=1)
    captured: list[list[str]] = []

    def _capture(cmd, log_path, env=None, append=False):
        captured.append(cmd)
        return _FakePopen()

    monkeypatch.setattr(exp_mod, "_spawn", _capture)
    launch_play_again(
        exp, "space-invaders", generation=1, episodes=5, seed=20_000, temperature=0.4,
    )
    assert len(captured) == 1
    cmd = captured[0]
    assert "play-again" in cmd
    assert "--generation" in cmd and cmd[cmd.index("--generation") + 1] == "1"
    assert "--episodes" in cmd and cmd[cmd.index("--episodes") + 1] == "5"
    assert "--temperature" in cmd and cmd[cmd.index("--temperature") + 1] == "0.4"
    assert "--seed" in cmd and cmd[cmd.index("--seed") + 1] == "20000"


def test_launch_play_again_shares_theater_lock(tmp_path: Path, monkeypatch):
    exp = create_experiment(tmp_path, "space-invaders", "pa-busy", knob_values={})
    _seed_run_config(exp, champion=1)
    monkeypatch.setattr(exp_mod, "_spawn", lambda cmd, log_path, env=None, append=False: _FakePopen())
    launch_theater(exp, "space-invaders", episodes=2)
    with pytest.raises(Busy):
        launch_play_again(exp, "space-invaders", generation=1)


def test_launch_play_again_rejects_missing_adapter(tmp_path: Path):
    exp = create_experiment(tmp_path, "space-invaders", "pa-noadapt", knob_values={})
    _seed_run_config(exp, champion=0)
    with pytest.raises(InvalidExperiment, match="no adapter"):
        launch_play_again(exp, "space-invaders", generation=9)


# --- HTTP route -----------------------------------------------------------


def test_http_play_again_launches_and_409s_on_second(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(exp_mod, "_spawn", lambda cmd, log_path, env=None, append=False: _FakePopen())
    exp = create_experiment(tmp_path, "space-invaders", "pa-http", knob_values={})
    _seed_run_config(exp, champion=2)

    with _ServerContext(tmp_path) as ctx:
        resp = ctx.post(
            "/api/experiments/pa-http/play-again",
            {"champion": True, "gen": None, "episodes": 3, "seed": 20000, "temperature": 0.2},
        )
        body = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 200, body
        assert body["generation"] == 2
        assert body["champion"] is True

        resp2 = ctx.post(
            "/api/experiments/pa-http/play-again",
            {"gen": 2, "champion": False, "episodes": 1},
        )
        body2 = json.loads(resp2.read().decode("utf-8"))
        assert resp2.status == 409
        assert "Busy" in body2["error"]


def test_http_play_again_validation_errors(tmp_path: Path):
    exp = create_experiment(tmp_path, "space-invaders", "pa-val", knob_values={})
    _seed_run_config(exp, champion=0)

    with _ServerContext(tmp_path) as ctx:
        resp = ctx.post(
            "/api/experiments/pa-val/play-again",
            {"champion": False, "gen": None},
        )
        body = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 400
        assert "gen" in body["error"].lower() or "provide" in body["error"].lower()

        resp2 = ctx.post(
            "/api/experiments/pa-val/play-again",
            {"champion": True},
        )
        body2 = json.loads(resp2.read().decode("utf-8"))
        assert resp2.status == 400
        assert "champion" in body2["error"].lower()

        resp3 = ctx.post(
            "/api/experiments/does-not-exist/play-again",
            {"gen": 0},
        )
        body3 = json.loads(resp3.read().decode("utf-8"))
        assert resp3.status == 400


def test_http_play_again_rejects_non_int_gen(tmp_path: Path):
    create_experiment(tmp_path, "space-invaders", "pa-badgen", knob_values={})
    with _ServerContext(tmp_path) as ctx:
        resp = ctx.post(
            "/api/experiments/pa-badgen/play-again",
            {"gen": "two", "champion": False},
        )
        body = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 400
        assert "gen" in body["error"]


def test_theater_play_side_serves_viewer(tmp_path: Path):
    create_experiment(tmp_path, "space-invaders", "pa-side", knob_values={})
    with _ServerContext(tmp_path) as ctx:
        resp = ctx.get("/theater/pa-side/play/")
        body = resp.read().decode("utf-8")
        assert resp.status == 200
        assert "SLM-RL — live play" in body


def test_page_has_play_again_ui():
    assert 'id="play-again-panel"' in PAGE
    assert 'data-card="play_again_button"' in PAGE
    assert "data-play=" in PAGE
    assert "/play-again" in PAGE
