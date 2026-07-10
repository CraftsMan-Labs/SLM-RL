"""GenerationRunner gate/registry/handoff flow with a fake backend + strategy
(no GPU, no model). Verifies: baseline caching, close() handoff ordering,
promote on improvement, reject on no-improvement, artifacts written."""

import json
from pathlib import Path

import pytest

from slm_rl.config.loader import load_run_config
from slm_rl.inference.base import GenOutput, GenParams, InferenceBackend
from slm_rl.training.base import TrainResult, TrainingStrategy

pytest.importorskip("pyarrow")  # run_generation consolidates to parquet


class FakeBackend(InferenceBackend):
    """Plays the winning move on a scripted schedule; records close() calls."""

    closed_count = 0

    def __init__(self, win_rate: float):
        self.win_rate = win_rate
        self._n = 0

    def generate(self, chats, params: GenParams):
        # deterministic: win `win_rate` fraction of the time by index
        self._n += 1
        return [GenOutput(text="ACTION: 1")]

    def load_adapter(self, path):
        pass

    def close(self):
        FakeBackend.closed_count += 1


def make_runner(tmp_path, monkeypatch, champ_primary, cand_primary, collapse=False):
    """Build a GenerationRunner whose eval returns scripted metrics and whose
    training is a no-op producing a dummy adapter."""
    from slm_rl import orchestrator
    import slm_rl.orchestrator.generation as gen

    cfg = load_run_config(game="mastermind", overrides={
        "run_id": "test", "home": str(tmp_path),
        "train": {"episodes_per_generation": 2},
    })

    # fake backend + strategy via the lazy factories the runner calls
    monkeypatch.setattr(gen, "create_backend", lambda *a, **k: FakeBackend(0.0))

    class FakeStrategy(TrainingStrategy):
        name = "fake"

        def train(self, dataset_path, out_dir, init_adapter=None):
            if collapse:
                return TrainResult(adapter_path=init_adapter, metrics={"entropy_collapsed": True})
            adapter = Path(out_dir) / "adapter"
            adapter.mkdir(parents=True, exist_ok=True)
            (adapter / "adapter_model.safetensors").write_text("weights")
            return TrainResult(adapter_path=adapter, metrics={"num_pairs": 5})

    monkeypatch.setattr(gen, "create_strategy", lambda *a, **k: FakeStrategy(cfg.train, "m"))

    runner = gen.GenerationRunner(cfg)

    # scripted eval: champion baseline vs candidate
    metrics_seq = iter([
        {"primary": champ_primary, "invalid_rate": 0.0, "intervention_rate": 0.0, "mean_entropy": None,
         "win_rate": champ_primary, "mean_score": 0.0, "episodes": 10},
        {"primary": cand_primary, "invalid_rate": 0.0, "intervention_rate": 0.0, "mean_entropy": None,
         "win_rate": cand_primary, "mean_score": 0.0, "episodes": 10},
    ])
    monkeypatch.setattr(runner, "_eval", lambda adapter, limit=None: next(metrics_seq))
    return runner


def test_promote_on_improvement(tmp_path, monkeypatch):
    runner = make_runner(tmp_path, monkeypatch, champ_primary=0.10, cand_primary=0.40)
    runner.ensure_baseline()
    FakeBackend.closed_count = 0
    m = runner.run_generation(1)

    assert m["gate"]["promoted"] is True
    assert runner.registry.champion == 1
    # rollout backend closed before training (the GPU handoff). Candidate eval
    # is mocked here, so only the rollout backend's close is observable.
    assert FakeBackend.closed_count == 1
    # artifacts
    assert runner.paths.metrics(1).exists()
    assert runner.paths.manifest(1).exists()
    assert (runner.paths.generation(1) / "eval" / "results.json").exists()


def test_reject_on_no_improvement(tmp_path, monkeypatch):
    runner = make_runner(tmp_path, monkeypatch, champ_primary=0.40, cand_primary=0.41)
    runner.ensure_baseline()
    m = runner.run_generation(1)

    assert m["gate"]["promoted"] is False
    assert runner.registry.champion == 0  # pointer never moved
    assert runner.registry.consecutive_failures == 1


def test_entropy_collapse_force_rejects_without_eval(tmp_path, monkeypatch):
    # candidate eval is scripted BETTER than champion, but collapse must
    # short-circuit before eval ever runs
    runner = make_runner(tmp_path, monkeypatch, champ_primary=0.10, cand_primary=0.90, collapse=True)
    runner.ensure_baseline()  # consumes the first scripted eval
    m = runner.run_generation(1)

    assert m["gate"]["promoted"] is False
    assert m["gate"]["reason"] == "train entropy collapsed"
    assert runner.registry.champion == 0
    assert m["eval"]["primary"] == 0.10  # champion metrics reused, no candidate eval


def test_baseline_cached(tmp_path, monkeypatch):
    runner = make_runner(tmp_path, monkeypatch, champ_primary=0.10, cand_primary=0.40)
    first = runner.ensure_baseline()
    # second call must read the cached file, not re-eval (eval iter is exhausted-safe)
    cached = json.loads((runner.paths.generation(0) / "eval" / "results.json").read_text())
    assert first == cached
