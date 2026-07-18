"""Plan 020: theater mounted inside the playground at `/theater/<name>/<side>/`,
the all-gens grid at `/gens/<name>/`, and the theater subprocess launcher's
busy-lock behavior. Same `_ServerContext` pattern as
tests/test_playground_watch.py (real ThreadingHTTPServer on port 0)."""

from __future__ import annotations

import http.client
import json
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from slm_rl.playground import experiments as exp_mod
from slm_rl.playground import server as pg_server_mod
from slm_rl.playground.experiments import Busy, create_experiment, launch_theater


def _write_rollout_record(path: Path, **fields) -> None:
    base = {
        "run_id": "pg-x", "generation": 0, "game": "space-invaders",
        "episode_id": "ep1", "step_idx": 0, "seed": 20000, "model_id": "m",
        "adapter_ref": None, "opponent_id": None,
        "prompt_messages": [{"role": "user", "content": "observe"}],
        "completion": "FIRE", "parsed_action": "FIRE", "legal_actions": ["FIRE"],
        "parse_status": "ok", "reward": 0.1, "shaped_reward": 0.1, "cum_reward": 0.1,
        "terminated": False, "truncated": False, "outcome": None,
        "state_hash": "abc", "monitor_flags": {}, "timestamp": "",
    }
    base.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(base) + "\n")


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

    def get(self, path: str) -> http.client.HTTPResponse:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", path)
        return conn.getresponse()

    def post(self, path: str, payload: dict) -> http.client.HTTPResponse:
        body = json.dumps(payload).encode("utf-8")
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
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


# --- /theater/<name>/<side>/ -----------------------------------------------


def test_theater_side_without_slash_redirects(tmp_path: Path):
    create_experiment(tmp_path, "space-invaders", "exp-a", knob_values={})
    with _ServerContext(tmp_path) as ctx:
        resp = ctx.get("/theater/exp-a/base")
        resp.read()
        assert resp.status == 301
        assert resp.getheader("Location") == "/theater/exp-a/base/"


def test_theater_base_side_serves_webui_page(tmp_path: Path):
    create_experiment(tmp_path, "space-invaders", "exp-b", knob_values={})
    with _ServerContext(tmp_path) as ctx:
        resp = ctx.get("/theater/exp-b/base/")
        body = resp.read().decode("utf-8")
        assert resp.status == 200
        assert "SLM-RL — live play" in body


def test_theater_events_streams_synthetic_records(tmp_path: Path):
    exp = create_experiment(tmp_path, "space-invaders", "exp-c", knob_values={})
    rollouts_dir = exp.run_dir / "theater" / "base" / "generations" / "gen_000" / "rollouts"
    f = rollouts_dir / "space-invaders.jsonl"
    _write_rollout_record(f, episode_id="ep1", step_idx=0)
    _write_rollout_record(f, episode_id="ep1", step_idx=1, terminated=True, outcome="win")

    with _ServerContext(tmp_path) as ctx:
        conn = http.client.HTTPConnection("127.0.0.1", ctx.port, timeout=5)
        conn.request("GET", "/theater/exp-c/base/events")
        resp = conn.getresponse()
        assert resp.status == 200
        assert resp.getheader("Content-Type") == "text/event-stream"

        seen = []
        deadline = time.monotonic() + 5.0
        while len(seen) < 2 and time.monotonic() < deadline:
            line = resp.fp.readline().decode("utf-8")
            if not line.startswith("data:"):
                continue
            payload = json.loads(line[len("data:"):].strip())
            seen.append(payload["step_idx"])
        conn.close()
        assert seen == [0, 1]


def test_theater_invalid_side_is_404(tmp_path: Path):
    create_experiment(tmp_path, "space-invaders", "exp-d", knob_values={})
    with _ServerContext(tmp_path) as ctx:
        resp = ctx.get("/theater/exp-d/not-a-side/")
        resp.read()
        assert resp.status == 404


def test_theater_invalid_names_are_404_and_prove_no_traversal(tmp_path: Path):
    secret = tmp_path / "secret.txt"
    secret.write_text("do not serve me", encoding="utf-8")

    invalid_names = ["..%2f..%2fsecret.txt", "UPPERCASE", "", "a" * 41]
    with _ServerContext(tmp_path) as ctx:
        for name in invalid_names:
            resp = ctx.get(f"/theater/{name}/base/")
            body = resp.read()
            assert resp.status == 404, f"expected 404 for {name!r}, got {resp.status}"
            assert b"do not serve me" not in body

        resp = ctx.get("/theater/../secret.txt")
        body = resp.read()
        assert resp.status == 404
        assert b"do not serve me" not in body


def test_theater_unknown_but_valid_name_is_404(tmp_path: Path):
    with _ServerContext(tmp_path) as ctx:
        resp = ctx.get("/theater/does-not-exist/base/")
        resp.read()
        assert resp.status == 404


# --- POST /api/experiments/<name>/theater ----------------------------------


class _FakePopen:
    pid = 1

    def poll(self):
        return None


def test_theater_launch_busy_returns_409(tmp_path: Path, monkeypatch):
    exp = create_experiment(tmp_path, "space-invaders", "exp-e", knob_values={})

    monkeypatch.setattr(exp_mod, "_spawn", lambda cmd, log_path, env=None, append=False: _FakePopen())
    launch_theater(exp, "space-invaders", episodes=2)

    with pytest.raises(Busy):
        launch_theater(exp, "space-invaders", episodes=2)


def test_http_post_theater_launches_and_second_call_409s(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(exp_mod, "_spawn", lambda cmd, log_path, env=None, append=False: _FakePopen())
    create_experiment(tmp_path, "space-invaders", "exp-f", knob_values={})

    with _ServerContext(tmp_path) as ctx:
        resp = ctx.post("/api/experiments/exp-f/theater", {})
        body = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 200
        assert body["name"] == "exp-f"

        resp2 = ctx.post("/api/experiments/exp-f/theater", {})
        resp2.read()
        assert resp2.status == 409


def test_http_post_theater_unknown_experiment_400s(tmp_path: Path):
    with _ServerContext(tmp_path) as ctx:
        resp = ctx.post("/api/experiments/does-not-exist/theater", {})
        resp.read()
        assert resp.status == 400


def test_http_stop_theater_stamps_failed_status(tmp_path: Path, monkeypatch):
    """Stop Theater must flip status.json off phase=base so the UI does not
    keep a partial episode labeled LIVE forever."""
    class _Alive(_FakePopen):
        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(exp_mod, "_spawn", lambda cmd, log_path, env=None, append=False: _Alive())
    monkeypatch.setattr(exp_mod, "_STOP_GRACE_SECONDS", 0.05)
    monkeypatch.setattr(exp_mod, "_terminate_proc", lambda proc: None)
    exp = create_experiment(tmp_path, "space-invaders", "exp-stop-stamp", knob_values={})
    status_path = exp.run_dir / "theater" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps({"phase": "base", "episode": 1, "episodes": 4}) + "\n",
        encoding="utf-8",
    )

    with _ServerContext(tmp_path) as ctx:
        resp = ctx.post("/api/experiments/exp-stop-stamp/theater", {"episodes": 2})
        resp.read()
        assert resp.status == 200
        resp = ctx.post(
            "/api/experiments/exp-stop-stamp/stop",
            {"kinds": ["theater"]},
        )
        body = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 200
        assert body["stopped"] == ["theater"]

    data = json.loads(status_path.read_text(encoding="utf-8"))
    assert data["phase"] == "failed"
    assert data["error"] == "stopped via UI"


# --- /api/experiments/<name>/theater-scores --------------------------------


def test_theater_scores_reflect_both_sides(tmp_path: Path):
    exp = create_experiment(tmp_path, "space-invaders", "exp-g", knob_values={})
    base_f = exp.run_dir / "theater" / "base" / "generations" / "gen_000" / "rollouts" / "space-invaders.jsonl"
    champ_f = exp.run_dir / "theater" / "champion" / "generations" / "gen_001" / "rollouts" / "space-invaders.jsonl"
    _write_rollout_record(base_f, episode_id="ep1", outcome="score:100", terminated=True)
    _write_rollout_record(champ_f, episode_id="ep1", outcome="score:300", terminated=True, generation=1)

    with _ServerContext(tmp_path) as ctx:
        resp = ctx.get("/api/experiments/exp-g/theater-scores")
        scores = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 200
        assert scores["base"]["mean_score"] == 100.0
        assert scores["champion"]["mean_score"] == 300.0


def test_theater_scores_missing_champion_side_omitted(tmp_path: Path):
    exp = create_experiment(tmp_path, "space-invaders", "exp-h", knob_values={})
    base_f = exp.run_dir / "theater" / "base" / "generations" / "gen_000" / "rollouts" / "space-invaders.jsonl"
    _write_rollout_record(base_f, episode_id="ep1", outcome="score:50", terminated=True)

    with _ServerContext(tmp_path) as ctx:
        resp = ctx.get("/api/experiments/exp-h/theater-scores")
        scores = json.loads(resp.read().decode("utf-8"))
        assert "base" in scores
        assert "champion" not in scores


# --- /gens/<name>/ ----------------------------------------------------------


def test_gens_page_without_slash_redirects(tmp_path: Path):
    create_experiment(tmp_path, "space-invaders", "exp-i", knob_values={})
    with _ServerContext(tmp_path) as ctx:
        resp = ctx.get("/gens/exp-i")
        resp.read()
        assert resp.status == 301
        assert resp.getheader("Location") == "/gens/exp-i/"


def test_gens_page_lists_one_panel_per_generation(tmp_path: Path):
    exp = create_experiment(tmp_path, "space-invaders", "exp-j", knob_values={})
    for gen in (0, 1, 2):
        (exp.run_dir / "generations" / f"gen_{gen:03d}" / "rollouts").mkdir(parents=True)

    with _ServerContext(tmp_path) as ctx:
        resp = ctx.get("/gens/exp-j/")
        body = resp.read().decode("utf-8")
        assert resp.status == 200
        for gen in (0, 1, 2):
            assert f"gen {gen}" in body
            assert f"/watch/exp-j/?gen={gen}" in body


# --- /watch/<name>/events?gen=N (the all-gens grid's actual filter) -------


def test_watch_events_gen_filter_only_yields_that_generation(tmp_path: Path):
    exp = create_experiment(tmp_path, "space-invaders", "exp-gen", knob_values={})
    f1 = exp.run_dir / "generations" / "gen_001" / "rollouts" / "a.jsonl"
    f2 = exp.run_dir / "generations" / "gen_002" / "rollouts" / "a.jsonl"
    _write_rollout_record(f1, episode_id="ep1", step_idx=0, generation=1)
    _write_rollout_record(f2, episode_id="ep2", step_idx=0, generation=2)

    with _ServerContext(tmp_path) as ctx:
        conn = http.client.HTTPConnection("127.0.0.1", ctx.port, timeout=5)
        conn.request("GET", "/watch/exp-gen/events?gen=2")
        resp = conn.getresponse()
        assert resp.status == 200

        seen = []
        deadline = time.monotonic() + 5.0
        while len(seen) < 1 and time.monotonic() < deadline:
            line = resp.fp.readline().decode("utf-8")
            if not line.startswith("data:"):
                continue
            payload = json.loads(line[len("data:"):].strip())
            seen.append(payload)
        conn.close()
        assert seen[0]["generation"] == 2
        assert seen[0]["episode_id"] == "ep2"


def test_gens_page_unknown_name_404s(tmp_path: Path):
    with _ServerContext(tmp_path) as ctx:
        resp = ctx.get("/gens/does-not-exist/")
        resp.read()
        assert resp.status == 404


def test_gens_page_no_generations_yet_is_200_with_empty_note(tmp_path: Path):
    create_experiment(tmp_path, "space-invaders", "exp-k", knob_values={})
    with _ServerContext(tmp_path) as ctx:
        resp = ctx.get("/gens/exp-k/")
        body = resp.read().decode("utf-8")
        assert resp.status == 200
        assert "no generations yet" in body


# --- regression: existing playground routes untouched -----------------------


def test_playground_page_contains_ab_and_gens_markup(tmp_path: Path):
    with _ServerContext(tmp_path) as ctx:
        resp = ctx.get("/")
        body = resp.read().decode("utf-8")
        assert resp.status == 200
        assert 'id="compare-panel"' in body
        assert 'data-compare' in body
        assert "/gens/" in body
