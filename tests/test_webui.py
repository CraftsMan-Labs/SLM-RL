"""Live-play viewer: tailer catch-up/live-append/malformed-line tolerance,
payload reduction, and an HTTP smoke test. Stdlib + pytest only — no model,
no GPU. Every server test binds port 0 and runs the server in a daemon
thread with a hard timeout so a hang here can't stall the suite."""

from __future__ import annotations

import http.client
import json
import threading
import time
from pathlib import Path

from slm_rl.webui.server import _feed, serve
from slm_rl.webui.tailer import iter_run_records, to_event


def _write_record(path: Path, **fields) -> None:
    base = {
        "run_id": "r1",
        "generation": 1,
        "game": "boxing",
        "episode_id": "ep1",
        "step_idx": 0,
        "seed": 0,
        "model_id": "m",
        "adapter_ref": None,
        "opponent_id": None,
        "prompt_messages": [{"role": "user", "content": "reveal?"}],
        "completion": "r0c0",
        "parsed_action": "r0c0",
        "legal_actions": ["r0c0"],
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


def test_catch_up_yields_existing_lines_in_generation_order(tmp_path: Path) -> None:
    run_dir = tmp_path / "run1"
    f1 = run_dir / "generations" / "gen_001" / "rollouts" / "a.jsonl"
    f2 = run_dir / "generations" / "gen_002" / "rollouts" / "b.jsonl"
    _write_record(f1, generation=1, episode_id="ep1", step_idx=0)
    _write_record(f1, generation=1, episode_id="ep1", step_idx=1)
    _write_record(f2, generation=2, episode_id="ep2", step_idx=0)

    stop = threading.Event()
    stop.set()  # exit after one catch-up pass
    recs = list(iter_run_records(run_dir, stop=stop))

    assert len(recs) == 3
    assert [r["generation"] for r in recs] == [1, 1, 2]
    assert [r["episode_id"] for r in recs] == ["ep1", "ep1", "ep2"]


def test_live_append_and_new_directory_pickup(tmp_path: Path) -> None:
    run_dir = tmp_path / "run1"
    f1 = run_dir / "generations" / "gen_001" / "rollouts" / "a.jsonl"
    _write_record(f1, episode_id="ep1", step_idx=0)

    stop = threading.Event()
    q: list[dict] = []
    lock = threading.Lock()

    def _consume() -> None:
        for rec in iter_run_records(run_dir, poll_interval=0.05, stop=stop):
            with lock:
                q.append(rec)

    thread = threading.Thread(target=_consume, daemon=True)
    thread.start()

    # Wait for the catch-up record to arrive.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with lock:
            if len(q) >= 1:
                break
        time.sleep(0.02)
    with lock:
        assert len(q) == 1

    # Append to the existing file.
    _write_record(f1, episode_id="ep1", step_idx=1)
    # Create a brand-new generation directory + file.
    f2 = run_dir / "generations" / "gen_002" / "rollouts" / "b.jsonl"
    _write_record(f2, episode_id="ep2", step_idx=0)

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with lock:
            if len(q) >= 3:
                break
        time.sleep(0.02)

    stop.set()
    thread.join(timeout=5.0)
    assert not thread.is_alive()

    with lock:
        assert len(q) == 3
        assert q[1]["episode_id"] == "ep1"
        assert q[1]["step_idx"] == 1
        assert q[2]["episode_id"] == "ep2"


def test_malformed_trailing_line_tolerated_then_yielded_once(tmp_path: Path) -> None:
    run_dir = tmp_path / "run1"
    path = run_dir / "generations" / "gen_001" / "rollouts" / "a.jsonl"
    path.parent.mkdir(parents=True)

    # Write a truncated JSON line with no trailing newline (writer mid-flush).
    full = {
        "run_id": "r1", "generation": 1, "game": "boxing",
        "episode_id": "ep1", "step_idx": 0, "seed": 0, "model_id": "m",
        "adapter_ref": None, "opponent_id": None,
        "prompt_messages": [{"role": "user", "content": "reveal?"}],
        "completion": "r0c0", "parsed_action": "r0c0",
        "legal_actions": ["r0c0"], "parse_status": "ok",
        "reward": 0.1, "shaped_reward": 0.1, "cum_reward": 0.1,
        "terminated": False, "truncated": False, "outcome": None,
        "state_hash": "abc", "monitor_flags": {}, "timestamp": "",
    }
    full_line = json.dumps(full)
    truncated = full_line[: len(full_line) // 2]  # cut mid-line, no newline
    path.write_text(truncated, encoding="utf-8")

    stop = threading.Event()
    q: list[dict] = []
    lock = threading.Lock()

    def _consume() -> None:
        for rec in iter_run_records(run_dir, poll_interval=0.05, stop=stop):
            with lock:
                q.append(rec)

    thread = threading.Thread(target=_consume, daemon=True)
    thread.start()

    # Give it a couple of polls to (not) pick up the truncated line.
    time.sleep(0.3)
    with lock:
        assert len(q) == 0

    # Complete that same line in place (the writer finishing its flush),
    # rather than appending a second line after it.
    path.write_text(full_line + "\n", encoding="utf-8")

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with lock:
            if len(q) >= 1:
                break
        time.sleep(0.02)

    stop.set()
    thread.join(timeout=5.0)
    assert not thread.is_alive()

    with lock:
        assert len(q) == 1
        assert q[0]["episode_id"] == "ep1"


def test_to_event_drops_prompt_messages_and_extracts_observed() -> None:
    rec = {
        "episode_id": "ep1",
        "generation": 1,
        "step_idx": 0,
        "prompt_messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "first turn"},
            {"role": "assistant", "content": "GUESS: RGBY"},
            {"role": "user", "content": "feedback: 1 black 1 white"},
        ],
        "completion": "GUESS: RGBY",
        "parsed_action": "RGBY",
        "legal_actions": ["RGBY"],
        "parse_status": "ok",
        "reward": 0.1,
        "cum_reward": 0.1,
        "terminated": False,
        "truncated": False,
        "outcome": None,
        "monitor_flags": {},
        "model_id": "m",
        "seed": 0,
    }
    event = to_event(rec)

    assert "prompt_messages" not in event
    assert event["observed"] == "feedback: 1 black 1 white"
    assert event["episode_id"] == "ep1"
    assert event["parsed_action"] == "RGBY"


def test_to_event_survives_missing_optional_fields() -> None:
    # Old-schema record: no monitor_flags, no prompt_messages, no outcome.
    rec = {"episode_id": "ep1", "step_idx": 0}
    event = to_event(rec)

    assert event["observed"] == ""
    assert event["monitor_flags"] is None
    assert event["outcome"] is None
    assert event["episode_id"] == "ep1"


def test_http_smoke_serves_page_and_streams_events(tmp_path: Path) -> None:
    run_dir = tmp_path / "run1"
    f1 = run_dir / "generations" / "gen_001" / "rollouts" / "a.jsonl"
    _write_record(f1, episode_id="ep1", step_idx=0)

    import socket
    from http.server import ThreadingHTTPServer

    from slm_rl.webui import server as server_mod

    handler_cls = server_mod._make_handler(run_dir)
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
        # Skip any keepalive comments (shouldn't happen this fast, but be safe).
        deadline = time.monotonic() + 5.0
        while line.startswith(":") and time.monotonic() < deadline:
            line = resp.fp.readline().decode("utf-8")
        assert line.startswith("data:")
        payload = json.loads(line[len("data:"):].strip())
        assert payload["episode_id"] == "ep1"
        conn.close()

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/nope")
        resp = conn.getresponse()
        assert resp.status == 404
        resp.read()
        conn.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5.0)


def test_serve_uses_threading_http_server_on_loopback(tmp_path: Path, monkeypatch) -> None:
    """`serve()` is the CLI entrypoint; verify it wires up ThreadingHTTPServer
    on the given host/port and actually calls serve_forever(), without
    leaving a real server thread running past the test (no exposed shutdown
    hook, so we substitute a server double instead of binding for real)."""
    calls = {}

    class FakeServer:
        def __init__(self, address, handler_cls):
            calls["address"] = address
            calls["handler_cls"] = handler_cls

        def serve_forever(self):
            calls["served"] = True

        def server_close(self):
            calls["closed"] = True

    monkeypatch.setattr(
        "slm_rl.webui.server.ThreadingHTTPServer", FakeServer
    )

    run_dir = tmp_path / "run1"
    run_dir.mkdir(parents=True)
    serve(run_dir, host="127.0.0.1", port=9999)

    assert calls["address"] == ("127.0.0.1", 9999)
    assert calls["served"] is True
    assert calls["closed"] is True


def test_feeder_exits_on_stop_even_when_queue_is_full() -> None:
    """A disconnected client stops draining the queue; the feeder must not
    block forever in q.put() (that leaks one daemon thread per reconnect).
    Fill a tiny bounded queue, let the feeder block on it, set stop, and
    assert the thread terminates promptly."""
    import itertools
    import queue as queue_mod

    stop = threading.Event()
    q: queue_mod.Queue[dict] = queue_mod.Queue(maxsize=1)
    # Endless record supply: without the stop check the feeder would spin
    # or block on the full queue forever.
    records = ({"step_idx": i} for i in itertools.count())

    feeder = threading.Thread(target=_feed, args=(records, q, stop), daemon=True)
    feeder.start()

    # Wait until the queue is full — the feeder is now blocked in put().
    deadline = time.monotonic() + 2.0
    while not q.full() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert q.full()

    stop.set()
    feeder.join(timeout=2.0)  # put timeout is 0.5s, so exit is well within 2s
    assert not feeder.is_alive()
