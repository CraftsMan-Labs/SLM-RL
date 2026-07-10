"""hf_push: repo layout + the loop-must-survive-failure guarantee (no network)."""

import warnings
from types import SimpleNamespace

import huggingface_hub

from slm_rl.datagen.hf_push import push_generation, try_push_generation


class FakeApi:
    calls = []

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
    create, upload = FakeApi.calls
    assert create[1] == "org/repo" and create[2]["repo_type"] == "dataset"
    assert upload[1]["path_in_repo"] == "run7/gen_003"
    assert "rollouts/*.jsonl" in upload[1]["allow_patterns"]


def test_try_push_swallows_errors(tmp_path, monkeypatch):
    class BrokenApi:
        def __init__(self):
            raise ConnectionError("hub down")

    monkeypatch.setattr(huggingface_hub, "HfApi", BrokenApi)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert try_push_generation("org/repo", "run7", 1, tmp_path) is None
    assert any("push failed" in str(x.message) for x in w)
