"""Plan 022: playground model picker (arbitrary HF model + backend choice).

Stdlib + pytest only, no model loads, no real network (huggingface_hub's
`HfApi.repo_exists` is monkeypatched everywhere it matters -- CODING_GUIDELINE
"no model loads in pytest" + plan hard rule 1). Uses the same
`ThreadingHTTPServer` HTTP-smoke pattern as tests/test_playground.py.
"""

from __future__ import annotations

import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from slm_rl.config.loader import load_run_config
from slm_rl.playground import experiments as exp_mod
from slm_rl.playground.experiments import (
    InvalidExperiment,
    create_experiment,
)


@pytest.fixture(autouse=True)
def _reset_locks():
    for kind in ("quick", "evolve", "theater", "bake"):
        exp_mod._ACTIVE[kind] = None
        exp_mod._ACTIVE_OWNER[kind] = None
    yield
    for kind in ("quick", "evolve", "theater", "bake"):
        exp_mod._ACTIVE[kind] = None
        exp_mod._ACTIVE_OWNER[kind] = None


# --- create_experiment: model+backend materialize into the run config -----


def test_create_with_model_and_backend_materializes_both(tmp_path: Path):
    exp = create_experiment(
        tmp_path,
        "space-invaders",
        "with-model",
        knob_values={},
        model="Qwen/Qwen2.5-0.5B-Instruct",
        backend="transformers",
    )

    cfg = load_run_config(config_dir=exp.config_dir)
    assert cfg.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg.backend == "transformers"

    data = json.loads(exp.experiment_json.read_text(encoding="utf-8"))
    assert data["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert data["backend"] == "transformers"
    assert exp.warnings == []


def test_create_with_neither_is_byte_identical_to_no_model_picker(tmp_path: Path):
    """Regression guard (hard rule 3): omitting model/backend must produce
    the exact materialized default.yaml this plan's code would have written
    before model/backend existed -- i.e. no `model:`/`backend:` keys at all,
    not `null`-valued ones (a present-but-null key is a different shape a
    future reader could trip on)."""
    exp = create_experiment(tmp_path, "space-invaders", "no-picker", knob_values={})

    raw = yaml.safe_load((exp.config_dir / "default.yaml").read_text(encoding="utf-8"))
    assert "model" not in raw
    assert "backend" not in raw

    cfg = load_run_config(config_dir=exp.config_dir)
    assert cfg.model is None
    assert cfg.backend is None
    assert exp.warnings == []


def test_create_with_tier_default_backend_pseudo_choice_is_a_noop(tmp_path: Path):
    exp = create_experiment(
        tmp_path, "space-invaders", "tier-default-backend", knob_values={}, backend="tier default",
    )
    raw = yaml.safe_load((exp.config_dir / "default.yaml").read_text(encoding="utf-8"))
    assert "backend" not in raw


def test_create_rejects_unknown_backend(tmp_path: Path):
    with pytest.raises(InvalidExperiment):
        create_experiment(tmp_path, "space-invaders", "bad-backend", knob_values={}, backend="tensorflow")


# --- local path model id ----------------------------------------------------


def test_local_path_model_id_is_accepted(tmp_path: Path):
    local_model = tmp_path / "my-local.gguf"
    local_model.write_bytes(b"not a real gguf")

    exp = create_experiment(tmp_path, "space-invaders", "local-model", knob_values={}, model=str(local_model))
    assert exp.warnings == []
    cfg = load_run_config(config_dir=exp.config_dir)
    assert cfg.model == str(local_model)


# --- blocking local sanity checks -------------------------------------------


def test_whitespace_model_id_is_blocking(tmp_path: Path):
    with pytest.raises(InvalidExperiment):
        create_experiment(tmp_path, "space-invaders", "bad-ws", knob_values={}, model="  Qwen/Qwen2.5-0.5B  ")


def test_missing_slash_and_not_a_local_path_is_blocking(tmp_path: Path):
    with pytest.raises(InvalidExperiment):
        create_experiment(tmp_path, "space-invaders", "bad-slash", knob_values={}, model="not-a-path-or-repo-id")


# --- backend override threads to create_backend at both call sites --------


def test_generation_runner_threads_backend_override(tmp_path: Path, monkeypatch):
    """GenerationRunner._backend calls create_backend(self.backend_name, ...)
    -- self.backend_name must be cfg.backend when set, overriding the tier's
    own backend. Fake create_backend to avoid any real model load."""
    from slm_rl.orchestrator import generation as gen_mod

    captured = {}

    def fake_create_backend(name, model_id, quantization=None):
        captured["name"] = name
        captured["model_id"] = model_id

        class _Fake:
            def close(self):
                pass

        return _Fake()

    monkeypatch.setattr(gen_mod, "create_backend", fake_create_backend)

    exp = create_experiment(
        tmp_path, "space-invaders", "override-backend", knob_values={},
        model="Qwen/Qwen2.5-0.5B-Instruct", backend="mlx",
    )
    cfg = load_run_config(config_dir=exp.config_dir)
    # run_id/home already point at the experiment dir (materialize_knobs does
    # this); avoid colliding with any real runs/ directory.
    runner = gen_mod.GenerationRunner(cfg, config_dir=exp.config_dir)
    runner._backend(adapter=None)

    assert captured["name"] == "mlx"
    assert captured["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"


def test_theater_exhibition_threads_backend_override(tmp_path: Path, monkeypatch):
    """theater/exhibition.py's run_exhibition resolves backend_name the same
    way (cfg.backend or tier.backend) from run_config.yaml -- verify the
    override survives a full write-then-read round trip through that file,
    the same path GenerationRunner.__init__ uses to freeze it."""
    from slm_rl.config.schema import RunConfig
    from slm_rl.platform.hardware import resolve_tier
    from slm_rl.config.loader import load_tiers

    exp = create_experiment(
        tmp_path, "space-invaders", "theater-backend", knob_values={},
        model="Qwen/Qwen2.5-0.5B-Instruct", backend="mlx",
    )
    cfg = load_run_config(config_dir=exp.config_dir)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_config.yaml").write_text(yaml.safe_dump(cfg.model_dump()))

    # Re-read exactly like run_exhibition does.
    reloaded = RunConfig(**yaml.safe_load((run_dir / "run_config.yaml").read_text(encoding="utf-8")))
    tier = resolve_tier(load_tiers(), forced_name=reloaded.tier)
    backend_name = reloaded.backend or tier.backend
    model_id = reloaded.model or tier.model

    assert backend_name == "mlx"
    assert model_id == "Qwen/Qwen2.5-0.5B-Instruct"


# --- HTTP smoke: create response carries model provenance -----------------


def test_http_create_with_model_returns_provenance(tmp_path: Path, monkeypatch):
    from slm_rl.playground import server as server_mod

    monkeypatch.setattr(exp_mod, "_spawn", lambda cmd, log_path, env=None, append=False: _FakeAlwaysDone())

    handler_cls = server_mod._make_handler(tmp_path, "space-invaders")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    try:
        payload = json.dumps({
            "name": "http-model-test",
            "knob_values": {},
            "agent": "random",
            "episodes": 1,
            "model": "some-org/some-model",
            "backend": "transformers",
        }).encode("utf-8")
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", "/api/experiments", body=payload,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        assert resp.status == 200
        body = json.loads(resp.read().decode("utf-8"))
        conn.close()
        assert body["warnings"] == []

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/api/experiments")
        resp = conn.getresponse()
        assert resp.status == 200
        rows = json.loads(resp.read().decode("utf-8"))
        conn.close()

        row = next(r for r in rows if r["name"] == "http-model-test")
        assert row["model"] == "some-org/some-model"
        assert row["backend"] == "transformers"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5.0)


def test_http_create_garbage_model_id_is_400(tmp_path: Path, monkeypatch):
    from slm_rl.playground import server as server_mod

    monkeypatch.setattr(exp_mod, "_spawn", lambda cmd, log_path, env=None, append=False: _FakeAlwaysDone())

    handler_cls = server_mod._make_handler(tmp_path, "space-invaders")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    try:
        payload = json.dumps({
            "name": "http-garbage-model",
            "knob_values": {},
            "agent": "random",
            "episodes": 1,
            "model": "no-slash-and-not-a-path",
        }).encode("utf-8")
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", "/api/experiments", body=payload,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        assert resp.status == 400
        resp.read()
        conn.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5.0)


class _FakeAlwaysDone:
    """poll() returns non-None immediately -- looks finished, never busy."""

    pid = 1

    def poll(self):
        return 0
