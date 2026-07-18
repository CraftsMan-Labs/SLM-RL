"""Plan 021: publish flow (dataset + champion adapter + model card) with
huggingface_hub fully mocked -- no real network calls. Mirrors
tests/test_hf_push.py's FakeApi pattern."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slm_rl.datagen.hf_publish import publish_experiment
from slm_rl.orchestrator.registry import ModelRegistry


def _write_champion_run(run_dir: Path, champ_gen: int = 1) -> None:
    """Materialize a run dir with a champion generation: registry.json,
    generations/gen_NNN/{adapter/, metrics.json, MANIFEST.json}, plus a
    gen_000 to exercise the dataset side pushing multiple generations."""
    for gen in range(champ_gen + 1):
        gen_dir = run_dir / "generations" / f"gen_{gen:03d}"
        (gen_dir / "rollouts").mkdir(parents=True)
        (gen_dir / "rollouts" / "game.jsonl").write_text('{"episode_id": "e1"}\n', encoding="utf-8")
        if gen == champ_gen:
            adapter_dir = gen_dir / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_model.safetensors").write_bytes(b"fake-weights")
            (gen_dir / "MANIFEST.json").write_text(
                json.dumps({"base_model": "Qwen/Qwen2.5-0.5B-Instruct"}), encoding="utf-8"
            )
            (gen_dir / "metrics.json").write_text(
                json.dumps({
                    "eval": {"primary": 0.83, "win_rate": 0.6},
                    "gate": {"promoted": True, "reason": "beat champion"},
                }),
                encoding="utf-8",
            )

    registry = ModelRegistry(run_dir / "registry.json")
    if champ_gen > 0:
        registry.promote(champ_gen, "beat champion")


class FakeApi:
    calls: list = []

    def __init__(self, token=None):
        FakeApi.calls.append(("__init__", token))

    def create_repo(self, repo_id, **kwargs):
        FakeApi.calls.append(("create_repo", repo_id, kwargs))

    def upload_folder(self, **kwargs):
        FakeApi.calls.append(("upload_folder", kwargs))
        return SimpleNamespace(commit_url="https://hf.co/fake/model-commit")

    def upload_file(self, **kwargs):
        FakeApi.calls.append(("upload_file", kwargs))
        return SimpleNamespace(commit_url="https://hf.co/fake/card-commit")


@pytest.fixture(autouse=True)
def _reset_calls():
    FakeApi.calls = []
    yield
    FakeApi.calls = []


def _patch_hub(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)


def test_publish_uploads_adapter_and_model_card(tmp_path: Path, monkeypatch):
    _patch_hub(monkeypatch)
    run_dir = tmp_path / "pg-x"
    _write_champion_run(run_dir, champ_gen=1)

    # dataset side also goes through HfApi -- push_generation constructs its
    # own HfApi(token=...), patched by the same monkeypatch above.
    result = publish_experiment(
        token="hf_realtoken1234", username="ada", experiment="x", game="space-invaders", run_dir=run_dir,
    )

    assert result.model_repo == "ada/slm-rl-x"
    assert result.model_error is None

    upload_folder_calls = [c for c in FakeApi.calls if c[0] == "upload_folder"]
    model_upload = next(c for c in upload_folder_calls if c[1]["repo_id"] == "ada/slm-rl-x")
    assert model_upload[1]["path_in_repo"] == "adapter"

    upload_file_calls = [c for c in FakeApi.calls if c[0] == "upload_file"]
    assert len(upload_file_calls) == 1
    card_bytes = upload_file_calls[0][1]["path_or_fileobj"]
    card_text = card_bytes.decode("utf-8")
    assert "library_name: peft" in card_text
    assert "space-invaders" in card_text
    assert "Qwen/Qwen2.5-0.5B-Instruct" in card_text
    assert "PeftModel.from_pretrained" in card_text
    assert 'subfolder="adapter"' in card_text
    assert "1" in card_text  # champion generation


def test_publish_dataset_side_uses_data_repo_suffix(tmp_path: Path, monkeypatch):
    _patch_hub(monkeypatch)
    run_dir = tmp_path / "pg-x"
    _write_champion_run(run_dir, champ_gen=1)

    result = publish_experiment(
        token="hf_realtoken1234", username="ada", experiment="x", game="space-invaders", run_dir=run_dir,
    )
    assert result.dataset_repo == "ada/slm-rl-x-data"
    assert result.dataset_error is None

    create_repo_calls = [c for c in FakeApi.calls if c[0] == "create_repo"]
    dataset_creates = [c for c in create_repo_calls if c[1] == "ada/slm-rl-x-data"]
    assert len(dataset_creates) == 2  # one per generation (gen_000, gen_001)
    assert all(c[2]["repo_type"] == "dataset" for c in dataset_creates)


def test_champion_less_run_publishes_datasets_only(tmp_path: Path, monkeypatch):
    _patch_hub(monkeypatch)
    run_dir = tmp_path / "pg-y"
    _write_champion_run(run_dir, champ_gen=0)  # registry.champion stays 0, no adapter

    result = publish_experiment(
        token="hf_realtoken1234", username="ada", experiment="y", game="space-invaders", run_dir=run_dir,
    )
    assert result.model_repo is None
    assert result.model_error is None
    assert result.message is not None
    assert "no promoted champion" in result.message

    assert result.dataset_repo == "ada/slm-rl-y-data"
    assert result.dataset_error is None


def test_token_passed_explicitly_and_never_logged(tmp_path: Path, monkeypatch, capsys):
    _patch_hub(monkeypatch)
    run_dir = tmp_path / "pg-z"
    _write_champion_run(run_dir, champ_gen=1)
    token = "hf_supersecrettoken999"

    publish_experiment(
        token=token, username="ada", experiment="z", game="space-invaders", run_dir=run_dir,
    )

    # every HfApi(...) construction got the token explicitly; per-call
    # token= on create_repo/upload_* is redundant once the client is built
    # with the token (plan 025 G6) and is no longer passed.
    init_calls = [c for c in FakeApi.calls if c[0] == "__init__"]
    assert init_calls, "expected HfApi(token=...) to be constructed"
    assert all(c[1] == token for c in init_calls)

    captured = capsys.readouterr()
    assert token not in captured.out
    assert token not in captured.err


def test_partial_failure_model_ok_dataset_raises(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "pg-w"
    _write_champion_run(run_dir, champ_gen=1)

    class ModelOkDatasetBrokenApi(FakeApi):
        def create_repo(self, repo_id, **kwargs):
            if kwargs.get("repo_type") == "dataset":
                raise ConnectionError("hub unreachable")
            return super().create_repo(repo_id, **kwargs)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", ModelOkDatasetBrokenApi)

    result = publish_experiment(
        token="hf_realtoken1234", username="ada", experiment="w", game="space-invaders", run_dir=run_dir,
    )
    assert result.model_repo == "ada/slm-rl-w"
    assert result.model_error is None
    assert result.dataset_repo == "ada/slm-rl-w-data"
    assert result.dataset_error is not None
    assert "hub unreachable" in result.dataset_error


def test_partial_failure_dataset_ok_model_raises(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "pg-v"
    _write_champion_run(run_dir, champ_gen=1)

    class DatasetOkModelBrokenApi(FakeApi):
        def create_repo(self, repo_id, **kwargs):
            if kwargs.get("repo_type") == "model":
                raise ConnectionError("model repo creation failed")
            return super().create_repo(repo_id, **kwargs)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", DatasetOkModelBrokenApi)

    result = publish_experiment(
        token="hf_realtoken1234", username="ada", experiment="v", game="space-invaders", run_dir=run_dir,
    )
    assert result.model_error is not None
    assert "model repo creation failed" in result.model_error
    assert result.dataset_repo == "ada/slm-rl-v-data"
    assert result.dataset_error is None


def test_no_generations_reports_clearly(tmp_path: Path, monkeypatch):
    _patch_hub(monkeypatch)
    run_dir = tmp_path / "pg-empty"
    run_dir.mkdir(parents=True)
    ModelRegistry(run_dir / "registry.json")  # champion stays 0

    result = publish_experiment(
        token="hf_realtoken1234", username="ada", experiment="empty", game="space-invaders", run_dir=run_dir,
    )
    assert result.dataset_error is not None
    assert "no generations" in result.dataset_error
    assert result.model_repo is None
