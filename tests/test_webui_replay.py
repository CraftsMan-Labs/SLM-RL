"""Live game screen (plan 010): PNG encoder, deterministic frame replay,
and the /frames HTTP route. PNG tests always run (stdlib only); the
env-dependent ones skip entirely on machines without the [atari] extra
(pytest.importorskip, matching tests/test_space_invaders.py's pattern).
"""

from __future__ import annotations

import http.client
import json
import struct
import threading
import zlib
from pathlib import Path

import pytest

from slm_rl.webui.png import encode_rgb
from slm_rl.webui.replay import ReplayUnavailable, replay_frames

# ---------------------------------------------------------------------------
# PNG encoder
# ---------------------------------------------------------------------------


def _decode_png_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    chunks = []
    offset = 8
    while offset < len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        tag = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        crc = data[offset + 8 + length : offset + 12 + length]
        assert struct.unpack(">I", crc)[0] == zlib.crc32(tag + payload) & 0xFFFFFFFF
        chunks.append((tag, payload))
        offset += 12 + length
    return chunks


def test_png_signature_and_ihdr_dims():
    # 2x2 RGB image: red, green, blue, white.
    pixels = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])
    png = encode_rgb(pixels, width=2, height=2)
    chunks = _decode_png_chunks(png)
    tags = [t for t, _ in chunks]
    assert tags == [b"IHDR", b"IDAT", b"IEND"]

    ihdr = chunks[0][1]
    width, height, bit_depth, color_type, comp, filt, interlace = struct.unpack(
        ">IIBBBBB", ihdr
    )
    assert (width, height) == (2, 2)
    assert bit_depth == 8
    assert color_type == 2  # truecolor RGB
    assert (comp, filt, interlace) == (0, 0, 0)


def test_png_idat_zlib_decompressible_round_trip():
    pixels = bytes([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
    png = encode_rgb(pixels, width=2, height=2)
    chunks = _decode_png_chunks(png)
    idat = next(payload for tag, payload in chunks if tag == b"IDAT")
    raw = zlib.decompress(idat)
    # filter byte 0 + 2 rows of 2*3 bytes each
    assert raw[0] == 0
    assert raw[1:7] == pixels[0:6]
    assert raw[7] == 0
    assert raw[8:14] == pixels[6:12]


def test_png_rejects_mismatched_byte_count():
    with pytest.raises(ValueError):
        encode_rgb(b"\x00" * 5, width=2, height=2)


# ---------------------------------------------------------------------------
# Replay: fixtures shared by determinism / unavailable / HTTP tests
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _base_record(**fields) -> dict:
    base = {
        "run_id": "r1",
        "generation": 0,
        "game": "space-invaders",
        "episode_id": "ep1",
        "step_idx": 0,
        "seed": 0,
        "model_id": "m",
        "adapter_ref": None,
        "opponent_id": None,
        "prompt_messages": [],
        "completion": "",
        "parsed_action": "FIRE",
        "legal_actions": ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"],
        "parse_status": "ok",
        "reward": 0.0,
        "shaped_reward": 0.0,
        "cum_reward": 0.0,
        "terminated": False,
        "truncated": False,
        "outcome": None,
        "state_hash": "x",
        "monitor_flags": {},
        "timestamp": "",
    }
    base.update(fields)
    return base


def _make_space_invaders_records(run_dir: Path, episode_id: str = "ep1") -> None:
    """3 real decisions (matching the actual adapter's action_repeat=3 from
    configs/games/space-invaders.yaml) using real ALE action-meaning strings,
    so replay.py's action lookup succeeds."""
    actions = ["FIRE", "RIGHT", "LEFT"]
    records = [
        _base_record(episode_id=episode_id, step_idx=i, seed=0, parsed_action=a)
        for i, a in enumerate(actions)
    ]
    records[-1]["terminated"] = False
    records[-1]["truncated"] = True  # mark the episode finished
    path = run_dir / "generations" / "gen_000" / "rollouts" / "space-invaders.jsonl"
    _write_jsonl(path, records)


def test_replay_unavailable_for_mastermind(tmp_path: Path) -> None:
    run_dir = tmp_path / "run1"
    rec = _base_record(game="mastermind-easy", episode_id="ep-mm", parsed_action="RGBY")
    path = run_dir / "generations" / "gen_000" / "rollouts" / "mastermind.jsonl"
    _write_jsonl(path, [rec])

    with pytest.raises(ReplayUnavailable):
        list(replay_frames(run_dir, "ep-mm"))


def test_replay_missing_episode_yields_nothing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run1"
    _make_space_invaders_records(run_dir, episode_id="ep1")
    assert list(replay_frames(run_dir, "does-not-exist")) == []


def test_replay_determinism_and_frame_count(tmp_path: Path) -> None:
    pytest.importorskip("ale_py")
    pytest.importorskip("gymnasium")

    run_dir = tmp_path / "run1"
    _make_space_invaders_records(run_dir, episode_id="ep1")

    frames_a = list(replay_frames(run_dir, "ep1"))
    frames_b = list(replay_frames(run_dir, "ep1"))

    # 3 decisions x action_repeat=3 (configs/games/space-invaders.yaml) = 9
    # frames unless an early terminal cuts a decision short (not expected in
    # the first few frames of a fresh episode).
    assert len(frames_a) == 9
    assert frames_a == frames_b  # byte-identical: the core correctness claim

    for frame in frames_a:
        assert frame[:8] == b"\x89PNG\r\n\x1a\n"


def test_replay_stops_on_stop_event_and_closes_env(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("ale_py")
    pytest.importorskip("gymnasium")

    run_dir = tmp_path / "run1"
    _make_space_invaders_records(run_dir, episode_id="ep1")

    closed = {"value": False}
    import gymnasium as gym

    real_make = gym.make

    def _tracking_make(*args, **kwargs):
        env = real_make(*args, **kwargs)
        real_close = env.close

        def _close():
            closed["value"] = True
            real_close()

        env.close = _close
        return env

    monkeypatch.setattr(gym, "make", _tracking_make)

    stop = threading.Event()
    gen = replay_frames(run_dir, "ep1", stop=stop)
    first = next(gen)
    assert first[:8] == b"\x89PNG\r\n\x1a\n"
    stop.set()
    # Draining the generator after `stop` is set must terminate promptly
    # (the poll loop checks `stop` before/after each sleep) and close the env.
    list(gen)
    assert closed["value"] is True


# ---------------------------------------------------------------------------
# HTTP: /frames route
# ---------------------------------------------------------------------------


def _run_server(run_dir: Path):
    from http.server import ThreadingHTTPServer

    from slm_rl.webui.server import _make_handler

    handler_cls = _make_handler(run_dir)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread, port


def test_frames_missing_episode_returns_404(tmp_path: Path) -> None:
    run_dir = tmp_path / "run1"
    _make_space_invaders_records(run_dir, episode_id="ep1")
    httpd, thread, port = _run_server(run_dir)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/frames?episode=does-not-exist")
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 404
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


def test_frames_missing_extra_returns_501(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run1"
    _make_space_invaders_records(run_dir, episode_id="ep1")

    def _boom(run_dir, episode_id, stop=None):
        raise ImportError("no module named ale_py")
        yield  # pragma: no cover - makes this a generator function

    monkeypatch.setattr("slm_rl.webui.server.replay_frames", _boom)

    httpd, thread, port = _run_server(run_dir)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/frames?episode=ep1")
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 501
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


def test_frames_unavailable_game_returns_501(tmp_path: Path) -> None:
    run_dir = tmp_path / "run1"
    rec = _base_record(game="mastermind-easy", episode_id="ep-mm", parsed_action="RGBY")
    path = run_dir / "generations" / "gen_000" / "rollouts" / "mastermind.jsonl"
    _write_jsonl(path, [rec])

    httpd, thread, port = _run_server(run_dir)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/frames?episode=ep-mm")
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 501
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


def test_frames_http_streams_png_parts(tmp_path: Path) -> None:
    pytest.importorskip("ale_py")
    pytest.importorskip("gymnasium")

    run_dir = tmp_path / "run1"
    _make_space_invaders_records(run_dir, episode_id="ep1")

    httpd, thread, port = _run_server(run_dir)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("GET", "/frames?episode=ep1")
        resp = conn.getresponse()
        assert resp.status == 200
        assert "multipart/x-mixed-replace" in resp.getheader("Content-Type", "")
        body = resp.read(4000)
        assert b"PNG" in body
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
