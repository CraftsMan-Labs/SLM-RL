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


def scripted_metrics(primary):
    return {
        "primary": primary, "invalid_rate": 0.0, "intervention_rate": 0.0,
        "mean_entropy": None, "win_rate": primary, "mean_score": 0.0, "episodes": 10,
    }


def make_runner(tmp_path, monkeypatch, champ_primary, cand_primary, collapse=False,
                more_candidates=(), teacher_overrides=None):
    """Build a GenerationRunner whose eval returns scripted metrics and whose
    training is a no-op producing a dummy adapter."""
    from slm_rl import orchestrator
    import slm_rl.orchestrator.generation as gen

    cfg = load_run_config(game="mastermind", overrides={
        "run_id": "test", "home": str(tmp_path),
        "train": {"episodes_per_generation": 2},
        "teacher": teacher_overrides,
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

    # scripted eval: champion baseline, candidate, then any extras in order
    metrics_seq = iter(
        [scripted_metrics(champ_primary), scripted_metrics(cand_primary)]
        + [scripted_metrics(p) for p in more_candidates]
    )
    monkeypatch.setattr(
        runner, "_eval", lambda adapter, limit=None, pruner=None: next(metrics_seq)
    )
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
    runner = make_runner(tmp_path, monkeypatch, champ_primary=0.40, cand_primary=0.405)
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


def test_warm_start_teacher_generation(tmp_path, monkeypatch):
    runner = make_runner(
        tmp_path, monkeypatch, champ_primary=0.00, cand_primary=0.50,
        teacher_overrides={"warmstart_episodes": 3},
    )
    runner.ensure_baseline()
    FakeBackend.closed_count = 0
    m = runner.run_generation(1, teacher=True)

    assert m["gate"]["promoted"] is True  # normal gate semantics, teacher never in eval
    assert m["rollout"]["episodes"] == 3  # warmstart_episodes, not episodes_per_generation
    assert FakeBackend.closed_count == 0  # no inference backend for teacher rollout
    manifest = json.loads(runner.paths.manifest(1).read_text())
    assert manifest["strategy"] == "reject_sft"  # forced regardless of tier
    assert manifest["rollout_model"] == "teacher:mastermind_solver"
    rollout_text = next(runner.paths.rollouts(1).glob("*.jsonl")).read_text()
    assert "teacher:mastermind_solver" in rollout_text
    assert runner.registry.next_generation == 2  # evolve resumes normally after


def test_remediation_lr_only_floored_and_reset_on_promotion(tmp_path, monkeypatch):
    runner = make_runner(
        tmp_path, monkeypatch, champ_primary=0.40, cand_primary=0.40,
        more_candidates=(0.40, 0.90),
    )
    runner.ensure_baseline()
    lr0, eb0 = runner.cfg.train.learning_rate, runner.cfg.train.entropy_bonus

    runner.run_generation(1)  # reject #1: no remediation yet
    assert runner.cfg.train.learning_rate == lr0

    runner.run_generation(2)  # reject #2: halve LR, entropy_bonus untouched
    assert runner.cfg.train.learning_rate == lr0 / 2
    assert runner.cfg.train.entropy_bonus == eb0  # the 7.82-entropy overshoot fix

    runner.cfg.train.learning_rate = 1.5e-6  # next halving must hit the floor...
    runner.run_generation(3)  # ...except this one promotes -> reset instead
    assert runner.registry.champion == 3
    assert runner.cfg.train.learning_rate == lr0
    assert runner.cfg.train.entropy_bonus == eb0


def test_remediation_lr_floor(tmp_path, monkeypatch):
    runner = make_runner(
        tmp_path, monkeypatch, champ_primary=0.40, cand_primary=0.40,
        more_candidates=(0.40,),
    )
    runner.ensure_baseline()
    runner.cfg.train.learning_rate = 1.5e-6
    runner.run_generation(1)
    runner.run_generation(2)  # 2 consecutive failures -> remediate, floored
    assert runner.cfg.train.learning_rate == 1e-6


def test_eval_pruned_side_metric_never_gates(tmp_path, monkeypatch):
    # eval_pruned scripted BETTER than the candidate: the gate must still
    # reject on the candidate's own (pure, format-mode) numbers
    runner = make_runner(
        tmp_path, monkeypatch, champ_primary=0.40, cand_primary=0.40,
        more_candidates=(0.95,),
        teacher_overrides={"pruner": True, "eval_pruned_episodes": 5},
    )
    assert runner.pruner is not None
    runner.ensure_baseline()
    m = runner.run_generation(1)

    assert m["gate"]["promoted"] is False  # 0.95 never reached the gate
    assert m["eval_pruned"]["primary"] == 0.95
    assert m["eval"]["primary"] == 0.40


def test_no_eval_pruned_without_pruner(tmp_path, monkeypatch):
    runner = make_runner(tmp_path, monkeypatch, champ_primary=0.10, cand_primary=0.40)
    runner.ensure_baseline()
    m = runner.run_generation(1)
    assert "eval_pruned" not in m
