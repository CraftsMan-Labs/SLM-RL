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
from collections.abc import Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from slm_rl.webui.page import PAGE
from slm_rl.webui.tailer import iter_run_records, to_event

_KEEPALIVE_SECONDS = 15.0
# Bounded backpressure: catch-up on a large run reads from disk far faster
# than the socket drains; unbounded, the queue could balloon to the whole
# run's payload in memory (100k+ events on multi-generation runs). 1000
# events is a couple of MB at most.
_QUEUE_MAXSIZE = 1000


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


def _make_handler(run_dir: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "SLM-RL-watch/1"

        def log_message(self, fmt: str, *args: object) -> None:
            pass  # quiet; this is a local dev viewer

        def do_GET(self) -> None:  # noqa: N802 (stdlib method name)
            if self.path == "/":
                self._serve_page()
            elif self.path == "/events":
                self._serve_events()
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

    return Handler


def serve(run_dir: Path, host: str = "127.0.0.1", port: int = 8777) -> None:
    """Serve the live-play viewer for `run_dir` until interrupted."""
    handler_cls = _make_handler(Path(run_dir))
    httpd = ThreadingHTTPServer((host, port), handler_cls)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
