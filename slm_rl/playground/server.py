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
from urllib.parse import parse_qs, urlparse

from slm_rl.playground.experiments import (
    Busy,
    ExperimentDir,
    InvalidExperiment,
    LOG_KINDS,
    NotRunning,
    _NAME_RE,
    active_jobs_for,
    create_experiment,
    is_bake_busy,
    launch_bake,
    launch_evolve,
    launch_play_again,
    launch_rollout,
    launch_theater,
    resolve_play_again_generation,
    stop_experiment,
    tail_bake_log,
    tail_log,
    update_experiment_knobs,
)
from slm_rl.playground.knobs import (
    AGENT_CHOICES,
    DEFAULT_EPISODES,
    DEFAULT_SEED,
    MAX_EPISODES,
    knobs_schema,
)
from slm_rl.playground.page import PAGE
from slm_rl.playground.profile import InvalidProfile, load_profile, save_profile
from slm_rl.playground.reward_template import TEMPLATE
from slm_rl.playground.stats import experiment_stats, exhibition_scores
from slm_rl.webui.server import _parse_gen, serve_events, serve_frames, serve_viewer_page

_SIDES = ("base", "champion", "play")

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


def _write_text(handler: BaseHTTPRequestHandler, status: int, text: str) -> None:
    body = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/plain; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _resolve_default_game(game: str | None) -> str:
    """CLI `--game` override, else first registry entry. Used only for form
    preselect + baseline auto-row — each experiment carries its own game."""
    from slm_rl.games.registry import available_games

    games = available_games()
    if game is not None:
        return game
    if not games:
        # ponytail: empty registry (broken install); keep a string so the
        # server still boots and /api/games returns [].
        return "mastermind"
    return games[0]


def _make_handler(home: Path, default_game: str) -> type[BaseHTTPRequestHandler]:
    baseline_lock = threading.Lock()
    baseline_launched = {"done": False}
    # Publish is I/O-bound (uploads), not CPU-bound like rollout/evolve/
    # theater, so it runs inline in the request-handling thread (each HTTP
    # request already gets its own thread under ThreadingHTTPServer) rather
    # than a subprocess. One lock, keyed by experiment name, so a
    # double-click can't launch two uploads of the same experiment at once;
    # publishing two DIFFERENT experiments concurrently is fine (each is its
    # own upload, no shared mutable state) so this is per-name, not global
    # like the quick/evolve/theater locks.
    publish_locks_guard = threading.Lock()
    publish_locks: dict[str, bool] = {}

    def _ensure_baseline() -> None:
        with baseline_lock:
            if baseline_launched["done"]:
                return
            try:
                exp = create_experiment(home, default_game, _BASELINE_NAME, knob_values={})
                launch_rollout(
                    exp, default_game, agent="solver",
                    episodes=DEFAULT_EPISODES, seed=DEFAULT_SEED,
                )
            except Busy:
                return  # another experiment is running; retry on the next poll
            baseline_launched["done"] = True  # only mark done once actually launched

    def _jobs_for(name: str) -> list[str]:
        """In-memory job table + *.pid sidecars (survives watchfiles restarts)."""
        jobs = list(active_jobs_for(name))
        # ponytail: after API restart `_ACTIVE` is empty but CLI children may
        # still be alive — trust pid files next to evolve/theater/rollout logs.
        try:
            from slm_rl.playground.evolve_monitor import _read_pid, find_pid_path

            exp = ExperimentDir(name=name, path=home / "playground" / name, run_id=f"pg-{name}")
            if "evolve" not in jobs:
                if _read_pid(find_pid_path(home, f"pg-{name}")) is not None:
                    jobs.append("evolve")
                else:
                    pid_path = exp.log_path("evolve").with_suffix(".pid")
                    if _read_pid(pid_path) is not None:
                        jobs.append("evolve")
            if "theater" not in jobs:
                if _read_pid(exp.log_path("theater").with_suffix(".pid")) is not None:
                    jobs.append("theater")
            if "quick" not in jobs:
                if _read_pid(exp.log_path("rollout").with_suffix(".pid")) is not None:
                    jobs.append("quick")
        except Exception:  # noqa: BLE001 — status must stay read-only-safe
            pass
        return jobs

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
                provenance = _read_provenance(exp)
                jobs = _jobs_for(path.name)
                # Scoreboard "running" used to mean incomplete JSONL even after
                # the subprocess died. Overlay live jobs so Stop / status match
                # the real process table.
                status = "running" if jobs else (
                    "stopped" if stats.get("status") == "running" else stats.get("status")
                )
                rows.append({
                    "name": path.name,
                    **stats,
                    **provenance,
                    "status": status,
                    "active_jobs": jobs,
                })
        return rows

    def _read_provenance(exp: ExperimentDir) -> dict[str, Any]:
        """`{model, backend, game, knob_values}` from `experiment.json`.
        model/backend None means "tier default" (plan 022). game None means
        a pre-026 experiment that never recorded one — scoreboard shows "—".
        knob_values is {} when missing (create with defaults only). Never
        raises."""
        try:
            data = json.loads(exp.experiment_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {
                "model": None, "backend": None, "game": None, "knob_values": {},
                "dataset_url": None, "dqn_url": None, "adapter_url": None,
            }
        knobs = data.get("knob_values")
        return {
            "model": data.get("model"),
            "backend": data.get("backend"),
            "game": data.get("game"),
            "knob_values": knobs if isinstance(knobs, dict) else {},
            "dataset_url": data.get("dataset_url"),
            "dqn_url": data.get("dqn_url"),
            "adapter_url": data.get("adapter_url"),
        }

    def _experiment_game(exp: ExperimentDir) -> str:
        """Game persisted on the experiment, else process default (pre-026
        dirs / mid-create). Used by evolve/theater/publish so multi-game
        containers launch the right subprocess."""
        game = _read_provenance(exp).get("game")
        return game if isinstance(game, str) and game else default_game

    def _experiment(name: str) -> ExperimentDir | None:
        """Resolve `name` to its `ExperimentDir`, or None if `name` fails the
        same regex/path-traversal guard `validate_name` uses, or the
        experiment directory doesn't exist. Shared by /watch/, /theater/,
        and /gens/ -- one traversal guard for every route that takes a
        name from the URL."""
        if not _NAME_RE.match(name):
            return None
        exp_path = home / "playground" / name
        if not exp_path.is_dir():
            return None
        return ExperimentDir(name=name, path=exp_path, run_id=f"pg-{name}")

    def _theater_run_dir(name: str, side: str) -> Path | None:
        """Resolve `<home>/playground/<name>`'s `theater/<side>` dir for the
        `/theater/<name>/<side>/` routes. `side` is validated against the
        fixed {"base", "champion"} set (never taken as a raw path segment).
        A not-yet-exhibited (but otherwise valid) experiment is fine: the
        tailer polls until data appears."""
        if side not in _SIDES:
            return None
        exp = _experiment(name)
        return exp.run_dir / "theater" / side if exp is not None else None

    def _serve_run_viewer(
        handler: BaseHTTPRequestHandler,
        run_dir: Path | None,
        action: str | None,
        *,
        query: dict[str, list[str]] | None = None,
        generation: int | None = None,
    ) -> None:
        """Shared page|events|frames dispatch for /watch/ and /theater/."""
        if run_dir is None:
            handler.send_error(404)
            return
        if action is None:
            serve_viewer_page(handler)
        elif action == "events":
            serve_events(handler, run_dir, generation=generation)
        else:  # frames
            serve_frames(handler, run_dir, query or {})

    def _gens_list(name: str) -> list[int] | None:
        """Generation numbers present in `name`'s own run dir (not a theater
        side), sorted -- the all-gens grid's panel list. None if `name`
        doesn't resolve to a real experiment."""
        exp = _experiment(name)
        if exp is None:
            return None
        if not exp.run_dir.exists():
            return []
        gens = []
        for p in exp.run_dir.glob("generations/gen_*"):
            try:
                gens.append(int(p.name.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return sorted(gens)

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
                elif parsed.path == "/api/games":
                    from slm_rl.games.registry import available_games

                    _write_json(
                        self, 200,
                        {"games": available_games(), "default": default_game},
                    )
                elif parsed.path == "/api/knobs":
                    qs = parse_qs(parsed.query)
                    knob_game = (qs.get("game") or [None])[0] or default_game
                    _write_json(self, 200, knobs_schema(knob_game))
                elif parsed.path == "/api/experiments":
                    _write_json(self, 200, _list_experiments())
                elif parsed.path == "/api/reward-template":
                    _write_json(self, 200, {"template": TEMPLATE})
                elif parsed.path == "/api/profile":
                    profile = load_profile(home)
                    if profile is None:
                        self.send_error(404)
                        return
                    _write_json(self, 200, profile.masked())
                elif parsed.path == "/api/hardware":
                    # Plan 026 Phase E: detect_host + resolve_tier + presets.
                    # Reuses platform/hardware.py — never duplicates nvidia-smi.
                    from slm_rl.playground.presets import hardware_payload

                    _write_json(self, 200, hardware_payload())
                elif parsed.path == "/api/packs":
                    from slm_rl.packs import list_local_packs

                    _write_json(self, 200, {
                        "packs": list_local_packs(home),
                        "baking": is_bake_busy(),
                    })
                elif parsed.path == "/api/packs/log":
                    _write_text(self, 200, tail_bake_log(home))
                elif parsed.path == "/api/dqn/jobs":
                    from slm_rl.playground.dqn_monitor import list_dqn_jobs

                    _write_json(self, 200, {"jobs": list_dqn_jobs(home)})
                elif (
                    len(parts) == 4
                    and parts[:2] == ["api", "dqn"]
                    and parts[3] == "metrics"
                ):
                    from slm_rl.playground.dqn_monitor import job_metrics

                    try:
                        _write_json(self, 200, job_metrics(home, parts[2]))
                    except ValueError as exc:
                        _write_json(self, 400, {"error": str(exc)})
                elif (
                    len(parts) == 4
                    and parts[:2] == ["api", "dqn"]
                    and parts[3] == "log"
                ):
                    from slm_rl.playground.dqn_monitor import job_log

                    try:
                        _write_text(self, 200, job_log(home, parts[2]))
                    except ValueError as exc:
                        _write_json(self, 400, {"error": str(exc)})
                elif parsed.path == "/api/evolve/jobs":
                    from slm_rl.playground.evolve_monitor import list_evolve_jobs

                    _write_json(self, 200, {"jobs": list_evolve_jobs(home)})
                elif (
                    len(parts) == 4
                    and parts[:2] == ["api", "evolve"]
                    and parts[3] == "metrics"
                ):
                    from slm_rl.playground.evolve_monitor import job_metrics as evolve_metrics

                    try:
                        _write_json(self, 200, evolve_metrics(home, parts[2]))
                    except ValueError as exc:
                        _write_json(self, 400, {"error": str(exc)})
                elif (
                    len(parts) == 4
                    and parts[:2] == ["api", "evolve"]
                    and parts[3] == "log"
                ):
                    from slm_rl.playground.evolve_monitor import job_log as evolve_log

                    try:
                        _write_text(self, 200, evolve_log(home, parts[2]))
                    except ValueError as exc:
                        _write_json(self, 400, {"error": str(exc)})
                elif len(parts) == 2 and parts[0] == "watch" and not parsed.path.endswith("/"):
                    # No trailing slash: 301 so the viewer page's relative
                    # endpoint URLs ("events", "frames?...") resolve under
                    # /watch/<name>/ instead of /watch/.
                    self.send_response(301)
                    self.send_header("Location", f"/watch/{parts[1]}/")
                    self.end_headers()
                elif len(parts) == 2 and parts[0] == "watch":
                    exp = _experiment(parts[1])
                    _serve_run_viewer(self, exp.run_dir if exp else None, None)
                elif len(parts) == 3 and parts[0] == "watch" and parts[2] == "events":
                    exp = _experiment(parts[1])
                    _serve_run_viewer(
                        self, exp.run_dir if exp else None, "events",
                        generation=_parse_gen(parse_qs(parsed.query)),
                    )
                elif len(parts) == 3 and parts[0] == "watch" and parts[2] == "frames":
                    exp = _experiment(parts[1])
                    _serve_run_viewer(
                        self, exp.run_dir if exp else None, "frames",
                        query=parse_qs(parsed.query),
                    )
                # --- /theater/<name>/<side>/... -- same shape as /watch/,
                # one extra path segment for the side ("base" | "champion").
                # `parts[2] in _SIDES` gates the redirect itself (not just
                # the eventual lookup): a traversal segment landing in the
                # side slot (e.g. "/theater/../secret.txt") must 404
                # immediately rather than 301 to a path that only 404s on
                # the NEXT hop.
                elif (
                    len(parts) == 3 and parts[0] == "theater"
                    and parts[2] in _SIDES and not parsed.path.endswith("/")
                ):
                    self.send_response(301)
                    self.send_header("Location", f"/theater/{parts[1]}/{parts[2]}/")
                    self.end_headers()
                elif len(parts) == 3 and parts[0] == "theater":
                    _serve_run_viewer(self, _theater_run_dir(parts[1], parts[2]), None)
                elif len(parts) == 4 and parts[0] == "theater" and parts[3] == "events":
                    _serve_run_viewer(self, _theater_run_dir(parts[1], parts[2]), "events")
                elif len(parts) == 4 and parts[0] == "theater" and parts[3] == "frames":
                    _serve_run_viewer(
                        self, _theater_run_dir(parts[1], parts[2]), "frames",
                        query=parse_qs(parsed.query),
                    )
                elif len(parts) == 4 and parts[:2] == ["api", "experiments"] and parts[3] == "theater-scores":
                    name = parts[2]
                    exp = _experiment(name)
                    if exp is None:
                        self.send_error(404)
                        return
                    _write_json(self, 200, exhibition_scores(exp.run_dir / "theater"))
                # Live subprocess log tail (plan 026 Phase F): text/plain last
                # 64KiB of rollout.log / evolve.log / theater.log. Unknown
                # experiment → 404; known exp but missing log file → 200 "".
                elif len(parts) == 4 and parts[:2] == ["api", "experiments"] and parts[3] == "log":
                    name = parts[2]
                    exp = _experiment(name)
                    if exp is None:
                        self.send_error(404)
                        return
                    qs = parse_qs(parsed.query)
                    kind = (qs.get("kind") or ["rollout"])[0]
                    if kind not in LOG_KINDS:
                        _write_json(
                            self, 400,
                            {"error": f"kind must be one of {sorted(LOG_KINDS)}"},
                        )
                        return
                    _write_text(self, 200, tail_log(exp, kind))
                # --- /gens/<name>/ -- one viewer panel per generation
                # present in the experiment's OWN run (not a theater side).
                elif len(parts) == 2 and parts[0] == "gens" and not parsed.path.endswith("/"):
                    self.send_response(301)
                    self.send_header("Location", f"/gens/{parts[1]}/")
                    self.end_headers()
                elif len(parts) == 2 and parts[0] == "gens":
                    gens = _gens_list(parts[1])
                    if gens is None:
                        self.send_error(404)
                        return
                    self._serve_gens_page(parts[1], gens)
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
                elif len(parts) == 4 and parts[:2] == ["api", "experiments"] and parts[3] == "rollout":
                    self._launch_rollout(parts[2])
                elif len(parts) == 4 and parts[:2] == ["api", "experiments"] and parts[3] == "theater":
                    self._launch_theater(parts[2])
                elif len(parts) == 4 and parts[:2] == ["api", "experiments"] and parts[3] == "play-again":
                    self._launch_play_again(parts[2])
                elif parsed.path == "/api/profile":
                    self._save_profile()
                elif parsed.path == "/api/packs/bake":
                    self._launch_bake()
                elif len(parts) == 4 and parts[:2] == ["api", "experiments"] and parts[3] == "publish":
                    self._publish(parts[2])
                elif len(parts) == 4 and parts[:2] == ["api", "experiments"] and parts[3] == "stop":
                    self._stop_experiment(parts[2])
                elif len(parts) == 4 and parts[:2] == ["api", "experiments"] and parts[3] == "knobs":
                    self._update_knobs(parts[2])
                elif (
                    len(parts) == 4
                    and parts[:2] == ["api", "dqn"]
                    and parts[3] == "stop"
                ):
                    self._stop_dqn(parts[2])
                elif (
                    len(parts) == 4
                    and parts[:2] == ["api", "evolve"]
                    and parts[3] == "stop"
                ):
                    self._stop_evolve(parts[2])
                else:
                    self.send_error(404)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _stop_dqn(self, game: str) -> None:
            from slm_rl.playground.dqn_monitor import stop_dqn_job

            try:
                _write_json(self, 200, stop_dqn_job(home, game))
            except ValueError as exc:
                _write_json(self, 400, {"error": str(exc)})
            except FileNotFoundError as exc:
                _write_json(self, 404, {"error": str(exc)})
            except OSError as exc:
                _write_json(self, 500, {"error": str(exc)})

        def _stop_evolve(self, run_id: str) -> None:
            from slm_rl.playground.evolve_monitor import stop_evolve_job

            try:
                _write_json(self, 200, stop_evolve_job(home, run_id))
            except ValueError as exc:
                _write_json(self, 400, {"error": str(exc)})
            except FileNotFoundError as exc:
                _write_json(self, 404, {"error": str(exc)})
            except OSError as exc:
                _write_json(self, 500, {"error": str(exc)})

        def _launch_bake(self) -> None:
            body = _read_body(self)
            game = (body.get("game") or "").strip() or None
            all_games = bool(body.get("all"))
            episodes = int(body.get("episodes", 1000))
            dqn_decisions = int(body.get("dqn_decisions", 50_000))
            selection_quantile = float(body.get("selection_quantile", 0.25))
            device = (body.get("device") or "cpu").strip() or "cpu"
            push = (body.get("push") or "").strip() or None
            push_prefix = (body.get("push_prefix") or "").strip() or None
            if not all_games and not game:
                _write_json(self, 400, {"error": "pick a game or set all=true"})
                return
            if not 0.0 < selection_quantile <= 1.0:
                _write_json(
                    self, 400,
                    {"error": "selection_quantile must be in (0, 1]"},
                )
                return
            profile = load_profile(home)
            token = profile.hf_token if profile else None
            if (push or push_prefix) and not token:
                _write_json(
                    self, 400,
                    {"error": "HF token required to push packs — add one on the welcome screen"},
                )
                return
            # Resolve whoami so BLANK/your-org placeholders become the
            # attendee's real HF username before the bake subprocess starts.
            if (push or push_prefix) and profile is not None and token:
                from slm_rl.playground.profile import resolve_username
                from slm_rl.packs import resolve_push_prefix, resolve_push_repo

                try:
                    profile = resolve_username(home, profile)
                    if push:
                        push = resolve_push_repo(push, profile.hf_username)
                    if push_prefix:
                        push_prefix = resolve_push_prefix(
                            push_prefix, profile.hf_username,
                        )
                except ValueError as exc:
                    _write_json(self, 400, {"error": str(exc)})
                    return
                except Exception as exc:  # noqa: BLE001 — Hub whoami / network
                    _write_json(
                        self, 400,
                        {"error": f"could not resolve HF username for push: {exc}"},
                    )
                    return
            try:
                launch_bake(
                    home,
                    game=game,
                    all_games=all_games,
                    episodes=episodes,
                    dqn_decisions=dqn_decisions,
                    device=device,
                    push=push,
                    push_prefix=push_prefix,
                    token=token,
                    selection_quantile=selection_quantile,
                )
            except Busy as exc:
                _write_json(self, 409, {"error": str(exc)})
                return
            except InvalidExperiment as exc:
                _write_json(self, 400, {"error": str(exc)})
                return
            _write_json(
                self, 200,
                {"ok": True, "baking": True, "push": push, "push_prefix": push_prefix},
            )

        def _save_profile(self) -> None:
            from slm_rl.hf_auth import apply_hf_token

            body = _read_body(self)
            name = body.get("name", "")
            hf_token = body.get("hf_token") or None
            try:
                profile = save_profile(home, name=name, hf_token=hf_token)
            except InvalidProfile as exc:
                _write_json(self, 400, {"error": str(exc)})
                return
            apply_hf_token(profile.hf_token)
            _write_json(self, 200, profile.masked())

        def _publish(self, name: str) -> None:
            exp = _experiment(name)
            if exp is None:
                _write_json(self, 400, {"error": f"unknown experiment: {name!r}"})
                return
            profile = load_profile(home)
            if profile is None or not profile.hf_token:
                _write_json(
                    self, 409,
                    {"error": "no Hugging Face token on file — add one in the profile card to publish"},
                )
                return

            body = _read_body(self)
            repo_name = (body.get("repo_name") or "").strip() or None

            with publish_locks_guard:
                if publish_locks.get(name):
                    _write_json(self, 409, {"error": f"a publish for {name!r} is already running"})
                    return
                publish_locks[name] = True
            try:
                from slm_rl.playground.profile import resolve_username

                try:
                    profile = resolve_username(home, profile)
                except Exception as exc:  # noqa: BLE001 - hub/network error, must not crash the handler
                    # huggingface_hub error text never echoes the token itself
                    # (confirmed: it reports auth status like "Invalid user
                    # token", not the token value) -- safe to surface as-is.
                    _write_json(self, 400, {"error": f"could not verify Hugging Face token: {exc}"})
                    return
                if not profile.hf_username:
                    _write_json(
                        self, 400,
                        {"error": "could not resolve a Hugging Face username for the stored token"},
                    )
                    return

                from slm_rl.datagen.hf_publish import publish_experiment

                try:
                    result = publish_experiment(
                        token=profile.hf_token, username=profile.hf_username,
                        experiment=name, game=_experiment_game(exp), run_dir=exp.run_dir,
                        repo_name=repo_name,
                    )
                except ValueError as exc:
                    _write_json(self, 400, {"error": str(exc)})
                    return
                _write_json(self, 200, result.to_json())
            finally:
                with publish_locks_guard:
                    publish_locks[name] = False

        def _create_and_launch(self) -> None:
            body = _read_body(self)
            name = body.get("name", "")
            knob_values = body.get("knob_values", {}) or {}
            reward_code = body.get("reward_code")
            agent = body.get("agent", "solver")
            episodes = min(int(body.get("episodes", DEFAULT_EPISODES)), MAX_EPISODES)
            seed = int(body.get("seed", DEFAULT_SEED))
            model = body.get("model") or None
            backend = body.get("backend") or None
            # Per-experiment game (plan 026 Phase C); body omits → process default.
            game = body.get("game") or default_game

            if agent not in AGENT_CHOICES:
                _write_json(self, 400, {"error": f"agent must be one of {AGENT_CHOICES}"})
                return
            dqn_url = (body.get("dqn_url") or "").strip() or None
            dataset_url = (body.get("dataset_url") or "").strip() or None
            adapter_url = (body.get("adapter_url") or "").strip() or None
            try:
                exp = create_experiment(
                    home, game, name, knob_values, reward_code,
                    model=model, backend=backend, dqn_url=dqn_url,
                    dataset_url=dataset_url, adapter_url=adapter_url,
                )
            except InvalidExperiment as exc:
                _write_json(self, 400, {"error": str(exc)})
                return
            # Vue sends launch=false (create then Run from workspace). Legacy
            # playground + tests default launch=true.
            launch = body.get("launch", True)
            if launch:
                profile = load_profile(home)
                token = profile.hf_token if profile else None
                try:
                    launch_rollout(
                        exp, game, agent=agent, episodes=episodes, seed=seed,
                        token=token,
                    )
                except Busy as exc:
                    _write_json(self, 409, {"error": str(exc)})
                    return
                except InvalidExperiment as exc:
                    _write_json(self, 400, {"error": str(exc)})
                    return
            _write_json(
                self, 200,
                {"name": name, "run_id": exp.run_id, "game": game, "warnings": exp.warnings},
            )

        def _launch_rollout(self, name: str) -> None:
            body = _read_body(self)
            agent = body.get("agent", "solver")
            episodes = min(int(body.get("episodes", DEFAULT_EPISODES)), MAX_EPISODES)
            seed = int(body.get("seed", DEFAULT_SEED))
            if agent not in AGENT_CHOICES:
                _write_json(self, 400, {"error": f"agent must be one of {AGENT_CHOICES}"})
                return
            exp = ExperimentDir(name=name, path=home / "playground" / name, run_id=f"pg-{name}")
            if not exp.config_dir.exists():
                _write_json(self, 400, {"error": f"unknown experiment: {name!r}"})
                return
            profile = load_profile(home)
            token = profile.hf_token if profile else None
            try:
                launch_rollout(
                    exp, _experiment_game(exp), agent=agent, episodes=episodes, seed=seed,
                    token=token,
                )
            except Busy as exc:
                _write_json(self, 409, {"error": str(exc)})
                return
            except InvalidExperiment as exc:
                _write_json(self, 400, {"error": str(exc)})
                return
            _write_json(self, 200, {"name": name, "run_id": exp.run_id})

        def _update_knobs(self, name: str) -> None:
            exp = _experiment(name)
            if exp is None or not exp.config_dir.exists():
                _write_json(self, 404, {"error": f"unknown experiment: {name!r}"})
                return
            if "evolve" in _jobs_for(name):
                _write_json(
                    self, 409,
                    {"error": "Stop Evolve before changing knobs (config is read at launch)."},
                )
                return
            body = _read_body(self)
            knobs = body.get("knob_values")
            if not isinstance(knobs, dict) or not knobs:
                _write_json(self, 400, {"error": "knob_values must be a non-empty object"})
                return
            try:
                merged = update_experiment_knobs(exp, knobs)
            except InvalidExperiment as exc:
                _write_json(self, 400, {"error": str(exc)})
                return
            _write_json(self, 200, {"name": name, "knob_values": merged})

        def _stop_experiment(self, name: str) -> None:
            exp = _experiment(name)
            if exp is None or not exp.config_dir.exists():
                _write_json(self, 404, {"error": f"unknown experiment: {name!r}"})
                return
            body = _read_body(self)
            # Optional: {"kinds": ["theater"]} stops only theater; evolve keeps
            # collecting. Omit / empty = stop every active job for this project.
            raw_kinds = body.get("kinds")
            kinds = None
            if raw_kinds is not None:
                if not isinstance(raw_kinds, list) or not all(isinstance(k, str) for k in raw_kinds):
                    _write_json(self, 400, {"error": "kinds must be a list of strings"})
                    return
                kinds = raw_kinds
            try:
                stopped = stop_experiment(name, kinds=kinds)
            except NotRunning:
                # In-memory slot lost (API restart) but evolve.pid still alive.
                want_evolve = kinds is None or "evolve" in kinds
                if want_evolve and "evolve" in _jobs_for(name):
                    from slm_rl.playground.evolve_monitor import stop_evolve_job

                    try:
                        stop_evolve_job(home, f"pg-{name}")
                        stopped = ["evolve"]
                    except FileNotFoundError as exc:
                        _write_json(self, 409, {"error": str(exc)})
                        return
                else:
                    _write_json(self, 409, {"error": f"no active job for experiment {name!r}"})
                    return
            except ValueError as exc:
                _write_json(self, 400, {"error": str(exc)})
                return
            # Append a durable breadcrumb so the log UI shows why output ended.
            for kind in stopped:
                log_kind = "rollout" if kind == "quick" else kind
                try:
                    with exp.log_path(log_kind).open("a", encoding="utf-8") as f:
                        f.write(f"\n[playground] stopped via UI ({kind})\n")
                except OSError:
                    pass
            # Theater status.json otherwise stays phase=base forever → UI
            # shows unfinished episodes as LIVE with no process behind them.
            if "theater" in stopped:
                try:
                    from slm_rl.theater.exhibition import mark_theater_ui_stopped

                    mark_theater_ui_stopped(exp.run_dir / "theater")
                except OSError:
                    pass
            _write_json(self, 200, {"name": name, "stopped": stopped})

        def _launch_evolve(self, name: str) -> None:
            body = _read_body(self)
            generations = int(body.get("generations", 2))
            dataset_url = (body.get("dataset_url") or "").strip() or None
            dqn_url = (body.get("dqn_url") or "").strip() or None
            adapter_url = (body.get("adapter_url") or "").strip() or None
            exp = ExperimentDir(name=name, path=home / "playground" / name, run_id=f"pg-{name}")
            if not exp.config_dir.exists():
                _write_json(self, 400, {"error": f"unknown experiment: {name!r}"})
                return
            # Fall back to URLs saved at project create (New project form).
            prov = _read_provenance(exp)
            if not dataset_url:
                saved = prov.get("dataset_url")
                if isinstance(saved, str) and saved.strip():
                    dataset_url = saved.strip()
            if not dqn_url:
                saved = prov.get("dqn_url")
                if isinstance(saved, str) and saved.strip():
                    dqn_url = saved.strip()
            if not adapter_url:
                saved = prov.get("adapter_url")
                if isinstance(saved, str) and saved.strip():
                    adapter_url = saved.strip()
            # generations comes from the form (default 2). Pack/SFT import is
            # free and not counted — CLI advances next_generation after import.
            for label, url in (("dataset_url", dataset_url), ("adapter_url", adapter_url)):
                if not url:
                    continue
                try:
                    from slm_rl.packs import normalize_repo_id

                    normalize_repo_id(url)
                except ValueError as exc:
                    _write_json(self, 400, {"error": f"{label}: {exc}"})
                    return
            # Persist URLs into experiment yaml so resume/DIY see them
            if dataset_url or dqn_url or adapter_url:
                import yaml

                cfg_path = exp.config_dir / "default.yaml"
                data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                if dataset_url:
                    data["dataset_url"] = dataset_url
                if dqn_url:
                    data["dqn_url"] = dqn_url
                if adapter_url:
                    data["adapter_url"] = adapter_url
                cfg_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
            profile = load_profile(home)
            token = profile.hf_token if profile else None
            try:
                launch_evolve(
                    exp, _experiment_game(exp), generations,
                    dataset_url=dataset_url, dqn_url=dqn_url, adapter_url=adapter_url,
                    token=token,
                )
            except Busy as exc:
                _write_json(self, 409, {"error": str(exc)})
                return
            _write_json(self, 200, {"name": name, "run_id": exp.run_id})

        def _launch_theater(self, name: str) -> None:
            body = _read_body(self)
            # Default 4: Docker-CPU base side is slow; 10 eps often dies mid-run
            # (watchfiles reload / OOM) before champion ever starts.
            episodes = int(body.get("episodes", 4))
            seed_start = int(body.get("seed_start", 20_000))
            exp = ExperimentDir(name=name, path=home / "playground" / name, run_id=f"pg-{name}")
            if not exp.config_dir.exists():
                _write_json(self, 400, {"error": f"unknown experiment: {name!r}"})
                return
            profile = load_profile(home)
            token = profile.hf_token if profile else None
            try:
                launch_theater(
                    exp, _experiment_game(exp), episodes=episodes, seed_start=seed_start,
                    token=token,
                )
            except Busy as exc:
                _write_json(self, 409, {"error": str(exc)})
                return
            _write_json(self, 200, {"name": name, "run_id": exp.run_id})

        def _launch_play_again(self, name: str) -> None:
            # Plan 026 Phase G: optional post-train replay of one checkpoint.
            body = _read_body(self)
            exp = _experiment(name)
            if exp is None or not exp.config_dir.exists():
                _write_json(self, 400, {"error": f"unknown experiment: {name!r}"})
                return
            champion = bool(body.get("champion", False))
            gen_raw = body.get("gen", None)
            # JSON null → None; reject bool (JSON true would otherwise pass as 1).
            if gen_raw is not None and not isinstance(gen_raw, int):
                _write_json(self, 400, {"error": "gen must be an int or null"})
                return
            try:
                episodes = min(int(body.get("episodes", 10)), MAX_EPISODES)
                seed = int(body.get("seed", 20_000))
                temperature = float(body.get("temperature", 0.2))
                generation = resolve_play_again_generation(
                    exp, gen=gen_raw, champion=champion,
                )
                launch_play_again(
                    exp, _experiment_game(exp),
                    generation=generation, episodes=episodes,
                    seed=seed, temperature=temperature,
                )
            except (TypeError, ValueError) as exc:
                _write_json(self, 400, {"error": f"invalid play-again args: {exc}"})
                return
            except InvalidExperiment as exc:
                _write_json(self, 400, {"error": str(exc)})
                return
            except Busy as exc:
                _write_json(self, 409, {"error": str(exc)})
                return
            _write_json(
                self, 200,
                {
                    "name": name, "run_id": exp.run_id,
                    "generation": generation, "champion": champion,
                },
            )

        def _serve_page(self) -> None:
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_gens_page(self, name: str, gens: list[int]) -> None:
            from slm_rl.playground.gens_page import render_gens_page

            body = render_gens_page(name, gens).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve(
    home: Path | str,
    game: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8780,
) -> None:
    """Serve the workshop playground for `home` (a runs/ root) until
    interrupted. `game` is an optional default for form preselect / baseline
    only — attendees pick the game per experiment in the UI (plan 026)."""
    from slm_rl.hf_auth import apply_hf_token

    home_path = Path(home)
    # Make welcome-screen token ambient for in-process Hub downloads
    # (create_experiment → resolve_dqn, etc.) before any request runs.
    profile = load_profile(home_path)
    apply_hf_token(profile.hf_token if profile else None)
    default_game = _resolve_default_game(game)
    handler_cls = _make_handler(home_path, default_game)
    httpd = ThreadingHTTPServer((host, port), handler_cls)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
