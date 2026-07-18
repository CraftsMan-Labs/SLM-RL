"""Plan 021: /api/profile and /api/experiments/<name>/publish routes. Same
`_ServerContext` (real ThreadingHTTPServer on port 0) pattern as
tests/test_playground_theater.py. huggingface_hub is fully mocked; no
network calls."""

from __future__ import annotations

import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from slm_rl.playground import experiments as exp_mod
from slm_rl.playground import server as pg_server_mod
from slm_rl.playground.experiments import create_experiment
from slm_rl.playground.profile import save_profile
from slm_rl.orchestrator.registry import ModelRegistry


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


# --- GET /api/profile --------------------------------------------------


def test_profile_404_before_signup(tmp_path: Path):
    with _ServerContext(tmp_path) as ctx:
        resp = ctx.get("/api/profile")
        resp.read()
        assert resp.status == 404


def test_profile_get_masks_token(tmp_path: Path):
    save_profile(tmp_path, name="Ada", hf_token="hf_abcdefghijklmnop")
    with _ServerContext(tmp_path) as ctx:
        resp = ctx.get("/api/profile")
        body = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 200
        assert body["name"] == "Ada"
        assert body["has_token"] is True
        assert body["token_masked"] == "...mnop"
        assert "hf_abcdefghijklmnop" not in json.dumps(body)
        assert "hf_token" not in body


# --- POST /api/profile --------------------------------------------------


def test_post_profile_saves_and_roundtrips(tmp_path: Path):
    with _ServerContext(tmp_path) as ctx:
        resp = ctx.post("/api/profile", {"name": "Grace", "hf_token": "hf_xyzxyzxyz"})
        body = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 200
        assert body["name"] == "Grace"
        assert body["has_token"] is True

        resp2 = ctx.get("/api/profile")
        body2 = json.loads(resp2.read().decode("utf-8"))
        assert body2["name"] == "Grace"


def test_post_profile_name_only_is_valid(tmp_path: Path):
    with _ServerContext(tmp_path) as ctx:
        resp = ctx.post("/api/profile", {"name": "Grace"})
        body = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 200
        assert body["has_token"] is False


def test_post_profile_bad_token_prefix_400s(tmp_path: Path):
    with _ServerContext(tmp_path) as ctx:
        resp = ctx.post("/api/profile", {"name": "Grace", "hf_token": "not-a-token"})
        body = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 400
        assert "error" in body


def test_post_profile_empty_name_400s(tmp_path: Path):
    with _ServerContext(tmp_path) as ctx:
        resp = ctx.post("/api/profile", {"name": ""})
        resp.read()
        assert resp.status == 400


# --- page signup-card marker --------------------------------------------


def test_page_html_has_signup_card_marker(tmp_path: Path):
    with _ServerContext(tmp_path) as ctx:
        resp = ctx.get("/")
        body = resp.read().decode("utf-8")
        assert resp.status == 200
        assert 'id="signup-card"' in body


# --- POST /api/experiments/<name>/publish -------------------------------


class FakeApi:
    calls: list = []

    def __init__(self, token=None):
        FakeApi.calls.append(("__init__", token))

    def whoami(self, token=None):
        return {"name": "ada"}

    def create_repo(self, repo_id, **kwargs):
        FakeApi.calls.append(("create_repo", repo_id, kwargs))

    def upload_folder(self, **kwargs):
        FakeApi.calls.append(("upload_folder", kwargs))
        return SimpleNamespace(commit_url="https://hf.co/fake/model-commit")

    def upload_file(self, **kwargs):
        FakeApi.calls.append(("upload_file", kwargs))
        return SimpleNamespace(commit_url="https://hf.co/fake/card-commit")


@pytest.fixture(autouse=True)
def _reset_fake_api_calls():
    FakeApi.calls = []
    yield


def test_publish_without_token_returns_409(tmp_path: Path):
    create_experiment(tmp_path, "space-invaders", "exp-a", knob_values={})
    save_profile(tmp_path, name="Ada")  # no token
    with _ServerContext(tmp_path) as ctx:
        resp = ctx.post("/api/experiments/exp-a/publish", {})
        body = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 409
        assert "error" in body


def test_publish_unknown_experiment_400s(tmp_path: Path):
    save_profile(tmp_path, name="Ada", hf_token="hf_realtoken1234")
    with _ServerContext(tmp_path) as ctx:
        resp = ctx.post("/api/experiments/does-not-exist/publish", {})
        resp.read()
        assert resp.status == 400


def test_publish_username_resolution_failure_is_clean_400_not_a_crash(tmp_path: Path, monkeypatch):
    """resolve_username's whoami() call can fail (expired/invalid token,
    network hiccup) -- the route must catch it and respond with a normal
    JSON 400, never let the exception propagate and drop the connection."""
    import huggingface_hub

    class BrokenWhoamiApi(FakeApi):
        def whoami(self, token=None):
            raise huggingface_hub.errors.HfHubHTTPError("Invalid user token.")

    monkeypatch.setattr(huggingface_hub, "HfApi", BrokenWhoamiApi)

    create_experiment(tmp_path, "space-invaders", "exp-c", knob_values={})
    save_profile(tmp_path, name="Ada", hf_token="hf_badtoken1234")

    with _ServerContext(tmp_path) as ctx:
        resp = ctx.post("/api/experiments/exp-c/publish", {})
        body = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 400
        assert "error" in body
        assert "hf_badtoken1234" not in json.dumps(body)


def test_publish_no_champion_reports_datasets_only(tmp_path: Path, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

    exp = create_experiment(tmp_path, "space-invaders", "exp-b", knob_values={})
    gen_dir = exp.run_dir / "generations" / "gen_000" / "rollouts"
    gen_dir.mkdir(parents=True)
    (gen_dir / "game.jsonl").write_text('{"episode_id": "e1"}\n', encoding="utf-8")
    ModelRegistry(exp.run_dir / "registry.json")  # champion stays 0
    save_profile(tmp_path, name="Ada", hf_token="hf_realtoken1234")

    with _ServerContext(tmp_path) as ctx:
        resp = ctx.post("/api/experiments/exp-b/publish", {})
        body = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 200
        assert body["model_repo"] is None
        assert "no promoted champion" in body["message"]
        assert body["dataset_repo"] == "ada/slm-rl-exp-b-data"
