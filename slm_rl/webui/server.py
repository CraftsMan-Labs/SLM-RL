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
# Workshop UI may briefly overlap live-follow + Watch remounts (abort races),
# so leave headroom above the usual 1–2 visible screens.
_MAX_CONCURRENT_REPLAYS = 8
_replay_slots = threading.Semaphore(_MAX_CONCURRENT_REPLAYS)
# Unknown / wrong-gen episode ids must not block the HTTP handler forever on
# the live poller — return 404 once catch-up + this brief wait find nothing.
_FIRST_FRAME_TIMEOUT_SECONDS = 2.0


def _next_with_timeout(
    items: Iterable[Any],
    stop: threading.Event,
    timeout: float,
) -> Any:
    """Like `next(items)`, but set `stop` and raise StopIteration on timeout."""
    box: dict[str, Any] = {}

    def _worker() -> None:
        try:
            box["value"] = next(items)  # type: ignore[call-overload]
        except StopIteration:
            box["empty"] = True
        except Exception as exc:  # noqa: BLE001 — surfaced to caller thread
            box["exc"] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        stop.set()
        thread.join(2.0)
        raise StopIteration
    if "exc" in box:
        raise box["exc"]
    if box.get("empty"):
        raise StopIteration
    return box["value"]


def _feed(
    items: Iterable[Any],
    q: queue.Queue[Any],
    stop: threading.Event,
    *,
    drop_oldest: bool = False,
) -> None:
    """Move items into a bounded queue. Exits promptly once `stop` is set,
    even while blocked on a full queue (client gone, consumer not draining).
    When `drop_oldest` is True (frame streams), drop the oldest queued item
    on Full so a live screen catches up to "now" instead of playing back a
    backlog of stale frames."""
    for item in items:
        if stop.is_set():
            return
        while True:
            try:
                q.put(item, timeout=0.5)
                break
            except queue.Full:
                if stop.is_set():
                    return
                if drop_oldest:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass


def _parse_gen(query: dict[str, list[str]]) -> int | None:
    """`?gen=N` -> N, or None if absent/unparseable (falls back to
    unfiltered -- a malformed query string must never 500 a viewer page)."""
    values = query.get("gen") or []
    if not values:
        return None
    try:
        return int(values[0])
    except ValueError:
        return None


def serve_viewer_page(handler: BaseHTTPRequestHandler) -> None:
    """Serve the live-play viewer PAGE. Identical response whether mounted
    at `/` (standalone webui) or `/watch/<name>/` (playground) — the page's
    endpoint URLs are relative (webui/page.py), so they resolve against
    whichever path the browser fetched the page from."""
    body = PAGE.encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def serve_events(
    handler: BaseHTTPRequestHandler, run_dir: Path, generation: int | None = None
) -> None:
    """Stream `run_dir`'s rollout records as Server-Sent Events.

    `generation` (plan 020 all-gens grid) is additive and optional: omitted
    (the default), behavior is byte-identical to before this filter existed
    -- every existing caller (webui standalone, playground /watch/) passes
    nothing and keeps seeing every generation. When set, records whose
    `generation` field doesn't match are skipped server-side (never sent
    down the wire at all, not just hidden client-side) -- cheaper for the
    all-gens grid's N-panel case, and it means a stale/old-schema record
    missing `generation` is silently dropped rather than mis-attributed to
    the filtered gen.
    """
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "keep-alive")
    handler.end_headers()

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
                handler.wfile.write(b": keepalive\n\n")
                handler.wfile.flush()
                continue
            if generation is not None and rec.get("generation") != generation:
                continue
            payload = json.dumps(to_event(rec))
            handler.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
            handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        stop.set()
        feeder.join(timeout=2.0)


def serve_frames(
    handler: BaseHTTPRequestHandler,
    run_dir: Path,
    query: dict[str, list[str]],
    *,
    fps: float = 30.0,
) -> None:
    """Serve a multipart/x-mixed-replace PNG frame stream for one episode
    in `run_dir`, replaying it via gymnasium/ale-py (lazy import inside
    `replay_frames`; degrades to 501 when the [atari] extra is missing)."""
    episode_ids = query.get("episode") or []
    episode_id = episode_ids[0] if episode_ids else None
    if not episode_id:
        handler.send_error(404, "missing ?episode=")
        return
    generation = _parse_gen(query)
    # Optional per-request override: /frames?episode=...&fps=4
    if query.get("fps"):
        try:
            fps = float(query["fps"][0])
        except (TypeError, ValueError, IndexError):
            pass
    fps = max(0.5, min(fps, 60.0))

    if not _replay_slots.acquire(blocking=False):
        handler.send_error(503, "too many concurrent replay streams")
        return
    try:
        _stream_frames(
            handler, run_dir, episode_id, generation=generation, fps=fps,
        )
    finally:
        _replay_slots.release()


def _episode_has_records(
    run_dir: Path,
    episode_id: str,
    *,
    generation: int | None,
) -> bool:
    """Cheap catch-up peek (no ALE): True if any matching JSONL rows exist."""
    stop = threading.Event()
    stop.set()
    return (
        next(
            iter_run_records(
                run_dir,
                stop=stop,
                generation=generation,
                episode_id=episode_id,
                follow=False,
            ),
            None,
        )
        is not None
    )


def _stream_frames(
    handler: BaseHTTPRequestHandler,
    run_dir: Path,
    episode_id: str,
    *,
    generation: int | None = None,
    fps: float = 30.0,
) -> None:
    stop = threading.Event()
    # Slow watch: keep every decision frame (no drop) so the full episode
    # is visible. Fast/live watch: small queue + drop-oldest to stay current.
    slow = fps < 15.0
    qsize = 512 if slow else _FRAME_QUEUE_MAXSIZE
    drop_oldest = not slow
    q: queue.Queue[bytes] = queue.Queue(maxsize=qsize)
    frame_delay = 1.0 / fps

    # replay_frames is a generator function: nothing in its body
    # runs until the first `next()`, so ReplayUnavailable (raised
    # before its first `yield`) only surfaces there — pull one
    # frame eagerly, before committing to a 200 response, so a 501
    # (or a 404 for an unknown episode) is still possible.
    on_disk = _episode_has_records(
        run_dir, episode_id, generation=generation,
    )
    frame_iter = replay_frames(
        run_dir,
        episode_id,
        stop=stop,
        generation=generation,
        follow=True,
    )
    try:
        if on_disk:
            first_frame = next(frame_iter)
        else:
            # Still-writing episode: brief live wait, then 404.
            first_frame = _next_with_timeout(
                frame_iter, stop, _FIRST_FRAME_TIMEOUT_SECONDS,
            )
    except StopIteration:
        handler.send_error(404, "unknown episode")
        return
    except ReplayUnavailable as exc:
        handler.send_error(501, str(exc))
        return
    except ImportError as exc:
        # [atari] extra (gymnasium/ale-py) not installed — degrade
        # gracefully rather than 500ing (8GB tier may run without
        # it; frame replay is best-effort, never load-bearing).
        handler.send_error(501, f"atari extra not installed: {exc}")
        return

    q.put(first_frame)
    feeder = threading.Thread(
        target=_feed, args=(frame_iter, q, stop),
        kwargs={"drop_oldest": drop_oldest}, daemon=True,
    )
    feeder.start()
    try:
        handler.send_response(200)
        handler.send_header(
            "Content-Type",
            f"multipart/x-mixed-replace; boundary={_FRAME_BOUNDARY.decode()}",
        )
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()

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
            handler.wfile.write(part)
            handler.wfile.flush()
            # Pace playback here (not in replay.py, which stays a
            # pure generator). Default ~30fps; slower fps for demos.
            time.sleep(frame_delay)
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        stop.set()
        feeder.join(timeout=2.0)


def _make_handler(
    run_dir: Path, *, fps: float = 30.0,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "SLM-RL-watch/1"

        def log_message(self, fmt: str, *args: object) -> None:
            pass  # quiet; this is a local dev viewer

        def do_GET(self) -> None:  # noqa: N802 (stdlib method name)
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/":
                serve_viewer_page(self)
            elif parsed.path == "/events":
                serve_events(self, run_dir, generation=_parse_gen(query))
            elif parsed.path == "/frames":
                serve_frames(self, run_dir, query, fps=fps)
            else:
                self.send_error(404)

    return Handler


def serve(
    run_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8777,
    *,
    fps: float = 30.0,
) -> None:
    """Serve the live-play viewer for `run_dir` until interrupted."""
    handler_cls = _make_handler(Path(run_dir), fps=fps)
    httpd = ThreadingHTTPServer((host, port), handler_cls)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
