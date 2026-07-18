"""hf_push: repo layout + the loop-must-survive-failure guarantee (no network)."""

import warnings
from types import SimpleNamespace

import pytest

# Transitive dep of the [cuda] extra (transformers/datasets), deliberately
# not declared on its own: an 8GB/atari-only env won't have it, and the
# push-to-hub path is cuda-tier functionality anyway.
huggingface_hub = pytest.importorskip("huggingface_hub")

from slm_rl.datagen.hf_push import push_generation, try_push_generation


class FakeApi:
    calls = []

    def __init__(self, token=None):
        FakeApi.calls.append(("__init__", token))

    def create_repo(self, repo_id, **kwargs):
        FakeApi.calls.append(("create_repo", repo_id, kwargs))

    def upload_folder(self, **kwargs):
        FakeApi.calls.append(("upload_folder", kwargs))
        return SimpleNamespace(commit_url="https://hf.co/fake/commit")


def test_push_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    FakeApi.calls = []
    url = push_generation("org/repo", "run7", 3, tmp_path)
    assert url == "https://hf.co/fake/commit"
    _init, create, upload = FakeApi.calls
    assert create[1] == "org/repo" and create[2]["repo_type"] == "dataset"
    assert upload[1]["path_in_repo"] == "run7/gen_003"
    assert "rollouts/*.jsonl" in upload[1]["allow_patterns"]


def test_push_generation_passes_token_explicitly(tmp_path, monkeypatch):
    """Plan 021 token hygiene: an explicit token= must reach HfApi's
    constructor (per-call token= dropped after HfApi(token=...), plan 025 G6)."""
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    FakeApi.calls = []
    push_generation("org/repo", "run7", 0, tmp_path, token="hf_secrettoken1234")
    init_call = FakeApi.calls[0]
    assert init_call == ("__init__", "hf_secrettoken1234")


def test_try_push_swallows_errors(tmp_path, monkeypatch):
    class BrokenApi:
        def __init__(self):
            raise ConnectionError("hub down")

    monkeypatch.setattr(huggingface_hub, "HfApi", BrokenApi)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert try_push_generation("org/repo", "run7", 1, tmp_path) is None
    assert any("push failed" in str(x.message) for x in w)
