"""Plan 014: the live-play viewer mounted inside the playground server at
`/watch/<name>/`. Stdlib + pytest only -- drives `_make_handler` against a
real `ThreadingHTTPServer` on port 0, same pattern as
`tests/test_playground.py`'s HTTP smoke test. `/frames` is not exercised
here (needs ale-py + a replayable run; its logic is untouched and already
covered by `tests/test_webui_replay.py`)."""

from __future__ import annotations

import http.client
import json
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

from slm_rl.playground import server as pg_server_mod
from slm_rl.playground.experiments import create_experiment
from slm_rl.webui import server as webui_server_mod


def _write_rollout_record(path: Path, **fields) -> None:
    base = {
        "run_id": "pg-x",
        "generation": 0,
        "game": "space-invaders",
        "episode_id": "ep1",
        "step_idx": 0,
        "seed": 0,
        "model_id": "m",
        "adapter_ref": None,
        "opponent_id": None,
        "prompt_messages": [{"role": "user", "content": "observe"}],
        "completion": "FIRE",
        "parsed_action": "FIRE",
        "legal_actions": ["FIRE"],
        "parse_status": "ok",
        "reward": 0.1,
        "shaped_reward": 0.1,
        "cum_reward": 0.1,
        "terminated": False,
        "truncated": False,
        "outcome": None,
        "state_hash": "abc",
        "monitor_flags": {},
        "timestamp": "",
    }
    base.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(base) + "\n")


class _ServerContext:
    """Starts the playground handler on a real ThreadingHTTPServer (port 0)
    in a daemon thread, and stops it on exit -- lets tests use plain
    `http.client` calls without repeating the boilerplate."""

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


# --- Test 1: /watch/<name> (no trailing slash) -> 301 ----------------------


def test_watch_name_without_slash_redirects_with_trailing_slash(tmp_path: Path) -> None:
    create_experiment(tmp_path, "space-invaders", "exp-a", knob_values={})

    with _ServerContext(tmp_path) as ctx:
        resp = ctx.get("/watch/exp-a")
        resp.read()
        assert resp.status == 301
        assert resp.getheader("Location") == "/watch/exp-a/"


# --- Test 2: /watch/<name>/ -> 200, body is the webui PAGE ------------------


def test_watch_name_slash_serves_webui_page(tmp_path: Path) -> None:
    create_experiment(tmp_path, "space-invaders", "exp-b", knob_values={})

    with _ServerContext(tmp_path) as ctx:
        resp = ctx.get("/watch/exp-b/")
        body = resp.read().decode("utf-8")
        assert resp.status == 200
        # Distinctive marker from slm_rl/webui/page.py.
        assert "SLM-RL — live play" in body
        assert 'new EventSource("events")' in body


# --- Test 3: /watch/<name>/events streams synthetic rollout records --------


def test_watch_events_streams_synthetic_rollout_records(tmp_path: Path) -> None:
    exp = create_experiment(tmp_path, "space-invaders", "exp-c", knob_values={})
    rollouts_dir = exp.run_dir / "generations" / "gen_001" / "rollouts"
    f = rollouts_dir / "a.jsonl"
    _write_rollout_record(f, episode_id="ep1", step_idx=0)
    _write_rollout_record(f, episode_id="ep1", step_idx=1, terminated=True, outcome="win")

    with _ServerContext(tmp_path) as ctx:
        conn = http.client.HTTPConnection("127.0.0.1", ctx.port, timeout=5)
        conn.request("GET", "/watch/exp-c/events")
        resp = conn.getresponse()
        assert resp.status == 200
        assert resp.getheader("Content-Type") == "text/event-stream"

        seen_step_idxs = []
        deadline = time.monotonic() + 5.0
        while len(seen_step_idxs) < 2 and time.monotonic() < deadline:
            line = resp.fp.readline().decode("utf-8")
            if not line.startswith("data:"):
                continue
            payload = json.loads(line[len("data:"):].strip())
            assert payload["episode_id"] == "ep1"
            seen_step_idxs.append(payload["step_idx"])
        conn.close()

        assert seen_step_idxs == [0, 1]


# --- Test 4: unknown/invalid names -> 404, no traversal -------------------


def test_watch_unknown_but_valid_name_is_404(tmp_path: Path) -> None:
    with _ServerContext(tmp_path) as ctx:
        resp = ctx.get("/watch/does-not-exist/")
        resp.read()
        assert resp.status == 404


def test_watch_invalid_names_are_404_and_prove_no_traversal(tmp_path: Path) -> None:
    # A file outside <home>/playground/ that a traversal would try to reach.
    secret = tmp_path / "secret.txt"
    secret.write_text("do not serve me", encoding="utf-8")

    invalid_names = [
        "..%2f..%2fsecret.txt",  # encoded traversal
        "UPPERCASE",  # regex requires lowercase
        "",  # empty
        "a" * 41,  # over the 40-char cap
    ]
    with _ServerContext(tmp_path) as ctx:
        for name in invalid_names:
            resp = ctx.get(f"/watch/{name}/")
            body = resp.read()
            assert resp.status == 404, f"expected 404 for {name!r}, got {resp.status}"
            assert b"do not serve me" not in body

        # Literal (unencoded) traversal segments split into extra path
        # parts and never match the /watch/<name>/... routes at all.
        resp = ctx.get("/watch/../secret.txt")
        body = resp.read()
        assert resp.status == 404
        assert b"do not serve me" not in body


# --- Test 5: valid experiment, run dir absent -> page 200s, events connect -


def test_watch_page_and_events_ok_when_run_dir_not_yet_created(tmp_path: Path) -> None:
    # create_experiment materializes config/ but launch_rollout is what
    # creates run_dir/generations/ -- exercise the "not launched yet" gap.
    create_experiment(tmp_path, "space-invaders", "exp-d", knob_values={})

    with _ServerContext(tmp_path) as ctx:
        resp = ctx.get("/watch/exp-d/")
        resp.read()
        assert resp.status == 200

        conn = http.client.HTTPConnection("127.0.0.1", ctx.port, timeout=5)
        conn.request("GET", "/watch/exp-d/events")
        resp = conn.getresponse()
        assert resp.status == 200
        assert resp.getheader("Content-Type") == "text/event-stream"
        conn.close()


# --- Test 6: webui regression -- standalone handler unaffected -------------


def test_standalone_webui_handler_still_serves_root_and_events(tmp_path: Path) -> None:
    run_dir = tmp_path / "run1"
    f = run_dir / "generations" / "gen_001" / "rollouts" / "a.jsonl"
    _write_rollout_record(f, episode_id="ep1", step_idx=0)

    handler_cls = webui_server_mod._make_handler(run_dir)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/")
        resp = conn.getresponse()
        body = resp.read().decode("utf-8")
        assert resp.status == 200
        assert "SLM-RL" in body
        conn.close()

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/events")
        resp = conn.getresponse()
        assert resp.status == 200
        line = resp.fp.readline().decode("utf-8")
        deadline = time.monotonic() + 5.0
        while line.startswith(":") and time.monotonic() < deadline:
            line = resp.fp.readline().decode("utf-8")
        assert line.startswith("data:")
        payload = json.loads(line[len("data:"):].strip())
        assert payload["episode_id"] == "ep1"
        conn.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5.0)


# --- Test 7: playground PAGE regression -- watch link/iframe markup -------


def test_playground_page_contains_watch_link_and_iframe_markup(tmp_path: Path) -> None:
    with _ServerContext(tmp_path) as ctx:
        resp = ctx.get("/")
        body = resp.read().decode("utf-8")
        assert resp.status == 200
        assert 'id="watch-panel"' in body
        assert 'id="watch-frame"' in body
        assert "data-watch" in body
        assert "/watch/" in body
