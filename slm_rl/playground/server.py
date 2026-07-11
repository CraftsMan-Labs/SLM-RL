"""HTTP server for the workshop playground: stdlib `http.server` only (like
`slm_rl/webui/server.py`, whose ThreadingHTTPServer + closure-handler style
this imitates).

Read-WRITE surface (unlike webui, see package docstring): `POST
/api/experiments` and `POST /api/experiments/<name>/evolve` create files and
spawn subprocesses, but only under `<home>/playground/`. Binds 127.0.0.1 by
default -- a local workshop tool, not a public service. Executing
attendee-written Python (the reward hook) is by design; see the package
docstring's security model.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from slm_rl.playground.experiments import (
    Busy,
    ExperimentDir,
    InvalidExperiment,
    create_experiment,
    launch_evolve,
    launch_rollout,
    tail_log,
)
from slm_rl.playground.knobs import (
    AGENT_CHOICES,
    DEFAULT_EPISODES,
    DEFAULT_SEED,
    MAX_EPISODES,
    knobs_schema,
)
from slm_rl.playground.page import PAGE
from slm_rl.playground.reward_template import TEMPLATE
from slm_rl.playground.stats import experiment_stats

# Baseline is a synthetic scoreboard row (repo-default knobs, no reward
# code): run lazily the first time the page loads, guarded by the same
# quick-experiment lock as any other experiment (plan 013 design).
_BASELINE_NAME = "baseline"


def _read_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8")) if raw else {}


def _write_json(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _make_handler(home: Path, game: str) -> type[BaseHTTPRequestHandler]:
    baseline_lock = threading.Lock()
    baseline_launched = {"done": False}

    def _ensure_baseline() -> None:
        with baseline_lock:
            if baseline_launched["done"]:
                return
            try:
                exp = create_experiment(home, game, _BASELINE_NAME, knob_values={})
                launch_rollout(exp, game, agent="solver", episodes=DEFAULT_EPISODES, seed=DEFAULT_SEED)
            except Busy:
                return  # another experiment is running; retry on the next poll
            baseline_launched["done"] = True  # only mark done once actually launched

    def _list_experiments() -> list[dict[str, Any]]:
        _ensure_baseline()
        root = home / "playground"
        rows = []
        if root.exists():
            for path in sorted(root.iterdir()):
                if not path.is_dir():
                    continue
                exp = ExperimentDir(name=path.name, path=path, run_id=f"pg-{path.name}")
                stats = experiment_stats(exp.run_dir)
                rows.append({"name": path.name, **stats})
        return rows

    class Handler(BaseHTTPRequestHandler):
        server_version = "SLM-RL-playground/1"

        def log_message(self, fmt: str, *args: object) -> None:
            pass  # quiet; this is a local workshop tool

        def do_GET(self) -> None:  # noqa: N802 (stdlib method name)
            parsed = urlparse(self.path)
            parts = [p for p in parsed.path.split("/") if p]
            try:
                if parsed.path == "/":
                    self._serve_page()
                elif parsed.path == "/api/knobs":
                    _write_json(self, 200, knobs_schema(game))
                elif parsed.path == "/api/experiments":
                    _write_json(self, 200, _list_experiments())
                elif parsed.path == "/api/reward-template":
                    _write_json(self, 200, {"template": TEMPLATE})
                elif len(parts) == 4 and parts[:2] == ["api", "experiments"] and parts[3] == "log":
                    name = parts[2]
                    exp = ExperimentDir(name=name, path=home / "playground" / name, run_id=f"pg-{name}")
                    kind = "evolve" if (exp.path / "evolve.log").exists() else "rollout"
                    _write_json(self, 200, {"lines": tail_log(exp, kind)})
                else:
                    self.send_error(404)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self) -> None:  # noqa: N802 (stdlib method name)
            parsed = urlparse(self.path)
            parts = [p for p in parsed.path.split("/") if p]
            try:
                if parsed.path == "/api/experiments":
                    self._create_and_launch()
                elif len(parts) == 4 and parts[:2] == ["api", "experiments"] and parts[3] == "evolve":
                    self._launch_evolve(parts[2])
                else:
                    self.send_error(404)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _create_and_launch(self) -> None:
            body = _read_body(self)
            name = body.get("name", "")
            knob_values = body.get("knob_values", {}) or {}
            reward_code = body.get("reward_code")
            agent = body.get("agent", "solver")
            episodes = min(int(body.get("episodes", DEFAULT_EPISODES)), MAX_EPISODES)
            seed = int(body.get("seed", DEFAULT_SEED))

            if agent not in AGENT_CHOICES:
                _write_json(self, 400, {"error": f"agent must be one of {AGENT_CHOICES}"})
                return
            try:
                exp = create_experiment(home, game, name, knob_values, reward_code)
            except InvalidExperiment as exc:
                _write_json(self, 400, {"error": str(exc)})
                return
            try:
                launch_rollout(exp, game, agent=agent, episodes=episodes, seed=seed)
            except Busy as exc:
                _write_json(self, 409, {"error": str(exc)})
                return
            _write_json(self, 200, {"name": name, "run_id": exp.run_id})

        def _launch_evolve(self, name: str) -> None:
            body = _read_body(self)
            generations = int(body.get("generations", 3))
            exp = ExperimentDir(name=name, path=home / "playground" / name, run_id=f"pg-{name}")
            if not exp.config_dir.exists():
                _write_json(self, 400, {"error": f"unknown experiment: {name!r}"})
                return
            try:
                launch_evolve(exp, game, generations)
            except Busy as exc:
                _write_json(self, 409, {"error": str(exc)})
                return
            _write_json(self, 200, {"name": name, "run_id": exp.run_id})

        def _serve_page(self) -> None:
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve(home: Path | str, game: str, host: str = "127.0.0.1", port: int = 8780) -> None:
    """Serve the workshop playground for `home` (a runs/ root) until
    interrupted."""
    handler_cls = _make_handler(Path(home), game)
    httpd = ThreadingHTTPServer((host, port), handler_cls)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
