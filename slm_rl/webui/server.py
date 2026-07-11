"""HTTP server for the live-play viewer: stdlib `http.server` only.

Read-only observer (CODING_GUIDELINE invariant 5): the served `run_dir` is
fixed at process start, there is no query-driven file access, and the
tailer opens files read-only. Binds to 127.0.0.1 by default — a local
viewer, not a public service.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from slm_rl.webui.page import PAGE
from slm_rl.webui.replay import ReplayUnavailable, replay_frames
from slm_rl.webui.tailer import iter_run_records, to_event

_KEEPALIVE_SECONDS = 15.0
# Bounded backpressure: catch-up on a large run reads from disk far faster
# than the socket drains; unbounded, the queue could balloon to the whole
# run's payload in memory (100k+ events on multi-generation runs). 1000
# events is a couple of MB at most.
_QUEUE_MAXSIZE = 1000
# Frames are much bigger than text events (a 210x160 RGB PNG, tens of KB) —
# a small bound is enough backpressure; dropping the oldest queued frame
# (never the newest) is fine for a live screen, we always want to catch up
# to "now", not play back a backlog.
_FRAME_QUEUE_MAXSIZE = 8
_FRAME_BOUNDARY = b"slmrlframe"
# Local dev tool, not a public service: cap concurrent replay streams so one
# runaway client (or a page left open in a background tab) can't spin up
# unbounded ALE envs, which are the expensive resource here (not sockets).
_MAX_CONCURRENT_REPLAYS = 4
_replay_slots = threading.Semaphore(_MAX_CONCURRENT_REPLAYS)


def _feed(
    records: Iterable[dict[str, Any]],
    q: queue.Queue[dict[str, Any]],
    stop: threading.Event,
) -> None:
    """Move records into a bounded queue. Exits promptly once `stop` is set,
    even while blocked on a full queue (client gone, consumer not draining) —
    a bare `q.put(rec)` would block forever and leak the feeder thread."""
    for rec in records:
        while True:
            try:
                q.put(rec, timeout=0.5)
                break
            except queue.Full:
                if stop.is_set():
                    return


def _feed_frames(
    frames: Iterable[bytes],
    q: queue.Queue[bytes],
    stop: threading.Event,
) -> None:
    """Move PNG frames into a small bounded queue, dropping the oldest
    queued frame (never the newest) when the consumer falls behind — unlike
    `_feed`, a live screen wants to catch up to "now", not play back a
    backlog of stale frames. Exits promptly once `stop` is set."""
    for frame in frames:
        if stop.is_set():
            return
        while True:
            try:
                q.put(frame, timeout=0.5)
                break
            except queue.Full:
                if stop.is_set():
                    return
                try:
                    q.get_nowait()  # drop the oldest, then retry the put
                except queue.Empty:
                    pass


def _make_handler(run_dir: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "SLM-RL-watch/1"

        def log_message(self, fmt: str, *args: object) -> None:
            pass  # quiet; this is a local dev viewer

        def do_GET(self) -> None:  # noqa: N802 (stdlib method name)
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._serve_page()
            elif parsed.path == "/events":
                self._serve_events()
            elif parsed.path == "/frames":
                self._serve_frames(parse_qs(parsed.query))
            else:
                self.send_error(404)

        def _serve_page(self) -> None:
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_events(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            # iter_run_records() sleeps internally between polls, so it
            # can't hand control back to us on its own for a keepalive tick.
            # Run it on a feeder thread and pull from a queue with a
            # timeout instead, so idle periods still get keepalives.
            stop = threading.Event()
            q: queue.Queue[dict] = queue.Queue(maxsize=_QUEUE_MAXSIZE)

            feeder = threading.Thread(
                target=_feed,
                args=(iter_run_records(run_dir, stop=stop), q, stop),
                daemon=True,
            )
            feeder.start()
            try:
                while True:
                    try:
                        rec = q.get(timeout=_KEEPALIVE_SECONDS)
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    payload = json.dumps(to_event(rec))
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                stop.set()
                feeder.join(timeout=2.0)

        def _serve_frames(self, query: dict[str, list[str]]) -> None:
            episode_ids = query.get("episode") or []
            episode_id = episode_ids[0] if episode_ids else None
            if not episode_id:
                self.send_error(404, "missing ?episode=")
                return

            if not _replay_slots.acquire(blocking=False):
                self.send_error(503, "too many concurrent replay streams")
                return
            try:
                self._stream_frames(episode_id)
            finally:
                _replay_slots.release()

        def _stream_frames(self, episode_id: str) -> None:
            stop = threading.Event()
            q: queue.Queue[bytes] = queue.Queue(maxsize=_FRAME_QUEUE_MAXSIZE)

            # replay_frames is a generator function: nothing in its body
            # runs until the first `next()`, so ReplayUnavailable (raised
            # before its first `yield`) only surfaces there — pull one
            # frame eagerly, before committing to a 200 response, so a 501
            # (or a 404 for an unknown episode: no records -> empty
            # iterator) is still possible.
            frame_iter = replay_frames(run_dir, episode_id, stop=stop)
            try:
                first_frame = next(frame_iter)
            except StopIteration:
                self.send_error(404, "unknown episode")
                return
            except ReplayUnavailable as exc:
                self.send_error(501, str(exc))
                return
            except ImportError as exc:
                # [atari] extra (gymnasium/ale-py) not installed — degrade
                # gracefully rather than 500ing (8GB tier may run without
                # it; frame replay is best-effort, never load-bearing).
                self.send_error(501, f"atari extra not installed: {exc}")
                return

            q.put(first_frame)
            feeder = threading.Thread(
                target=_feed_frames, args=(frame_iter, q, stop), daemon=True
            )
            feeder.start()
            try:
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    f"multipart/x-mixed-replace; boundary={_FRAME_BOUNDARY.decode()}",
                )
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                while True:
                    try:
                        frame = q.get(timeout=_KEEPALIVE_SECONDS)
                    except queue.Empty:
                        if feeder.is_alive():
                            continue
                        try:
                            frame = q.get_nowait()  # closes a join/get race
                        except queue.Empty:
                            break  # replay finished and queue drained
                    part = (
                        b"--" + _FRAME_BOUNDARY + b"\r\n"
                        b"Content-Type: image/png\r\n"
                        b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                        + frame + b"\r\n"
                    )
                    self.wfile.write(part)
                    self.wfile.flush()
                    # Pace playback here (not in replay.py, which stays a
                    # pure generator): ~30fps is a natural watch speed.
                    time.sleep(1 / 30)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                stop.set()
                feeder.join(timeout=2.0)

    return Handler


def serve(run_dir: Path, host: str = "127.0.0.1", port: int = 8777) -> None:
    """Serve the live-play viewer for `run_dir` until interrupted."""
    handler_cls = _make_handler(Path(run_dir))
    httpd = ThreadingHTTPServer((host, port), handler_cls)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
