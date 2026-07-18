"""Evolve / SFT monitor: log parse + run discovery (no torch)."""

from __future__ import annotations

import json
from pathlib import Path

from slm_rl.playground.evolve_monitor import (
    _parse_phase,
    job_metrics,
    list_evolve_jobs,
    parse_crash_error,
    parse_run_plan,
    parse_train_loss_from_log,
    parse_train_progress,
)


SAMPLE = """
[evolve] run_id=boxing-sft-001 game=boxing generations=5 backend=transformers
[evolve] next generation=1 (will run until 5)
[evolve] gen 0 baseline: playing 100 frozen-suite episodes (boxing)
[evolve] gen 0 baseline: done primary=-0.420 (mean_score)
[packs] resolving BLANK/slm-rl-boxing
[evolve] gen 1: train start strategy=reject_sft
{'loss': 1.234, 'learning_rate': 0.0001, 'epoch': 0.05}
{'loss': 0.987, 'learning_rate': 0.0001, 'epoch': 0.10}
[evolve] gen 1: train done metrics={'num_pairs': 5000, 'loss': 0.5}
[evolve] gen 1: PROMOTED — baked pack adopted as RL initialization (not gated)
"""

GRPO_SAMPLE = """
[evolve] run_id=pg-x game=boxing generations=5
[evolve] next generation=2 (will run until 6)
[evolve] gen 2: train start strategy=grpo
  62%|██████▎   | 5/8 [02:24<01:26, 28.82s/it]
{'loss': -0.05, 'kl': 1.41, 'entropy': 2.345, 'reward': -0.7625, 'epoch': 0.18}
  75%|███████▌  | 6/8 [02:57<01:00, 30.20s/it]
"""


def test_parse_phase_empty_log_is_not_starting():
    assert _parse_phase("") == {"phase": "", "phase_generation": None}
    assert _parse_phase("   \n") == {"phase": "", "phase_generation": None}
    assert _parse_phase(SAMPLE)["phase"] == "promoted"


def test_parse_trl_loss_lines():
    rows = parse_train_loss_from_log(SAMPLE)
    assert len(rows) == 2
    assert rows[0]["loss"] == 1.234
    assert rows[1]["epoch"] == 0.1


def test_parse_crash_error_from_rich_traceback():
    log = (
        "[evolve] gen 2 gate eval: done primary=-1.000 (mean_score)\n"
        "╭───────────────────── Traceback (most recent call last) ──────────────────────╮\n"
        "│ /app/slm_rl/eval/gate.py:33 in decide                                        │\n"
        "╰──────────────────────────────────────────────────────────────────────────────╯\n"
        "KeyError: 'intervention_rate'\n"
    )
    assert parse_crash_error(log) == "KeyError: 'intervention_rate'"
    assert parse_crash_error("[evolve] gen 1: PROMOTED\n") is None


def test_parse_run_plan_and_train_progress():
    plan = parse_run_plan(SAMPLE)
    assert plan["target_generations"] == 5
    assert plan["start_generation"] == 1
    assert plan["end_generation"] == 5
    early = parse_run_plan(
        SAMPLE
        + "[evolve] gen 2: rejected — primary flat\n"
        + "[evolve] gen 3: rejected — primary flat\n"
        + "[evolve] early stop at gen 3: 2 consecutive rejects (limit 2)\n"
    )
    assert early["end_generation"] == 3
    assert early["target_generations"] == 3
    prog = parse_train_progress(GRPO_SAMPLE)
    assert prog["train_step"] == 6
    assert prog["train_total_steps"] == 8
    assert prog["train_kl"] == 1.41
    assert prog["train_entropy"] == 2.345
    assert prog["train_reward"] == -0.7625
    # After train start, stale progress from a prior gen must not leak.
    reset = parse_train_progress(
        "[evolve] gen 3: train start strategy=grpo\n"
        "  12%|█▎        | 1/8 [00:30<03:30, 30.00s/it]\n"
    )
    assert reset["train_step"] == 1
    assert reset["train_kl"] is None
    # Model load progress must not look like GRPO steps.
    load = parse_train_progress(
        "[evolve] gen 2: train start strategy=grpo\n"
        "Loading weights: 100%|██████████| 148/148 [00:00<00:00, 319.78it/s]\n"
    )
    assert load["train_step"] is None



def test_list_jobs_from_log_and_run_dir(tmp_path: Path):
    home = tmp_path / "runs"
    run = home / "boxing-sft-001"
    run.mkdir(parents=True)
    (run / "run_config.yaml").write_text("game: boxing\nmodel: LiquidAI/x\nbackend: transformers\n")
    g0 = run / "generations" / "gen_000" / "eval"
    g0.mkdir(parents=True)
    (g0 / "results.json").write_text(json.dumps({"primary": -0.42}))
    (run / "generations" / "gen_000" / "metrics.json").write_text(
        json.dumps({"eval": {"primary": -0.42}, "train": {}, "gate": {}}),
    )
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "evolve-boxing-sft-001.log").write_text(SAMPLE)
    # Use a PID that is almost certainly not alive (PID 1 always is on Unix).
    (logs / "evolve-boxing-sft-001.pid").write_text("2147483646\n")

    jobs = list_evolve_jobs(home)
    assert len(jobs) == 1
    assert jobs[0]["run_id"] == "boxing-sft-001"
    assert jobs[0]["phase"] == "promoted"
    assert jobs[0]["game"] == "boxing"
    assert jobs[0]["last_primary"] == -0.42

    m = job_metrics(home, "boxing-sft-001")
    assert len(m["train"]) == 2
    assert m["phase"] == "promoted"
    assert m["running"] is False
    assert "pid" in m
    assert m["target_generations"] == 5
    assert m["start_generation"] == 1
    assert m["end_generation"] == 5
    assert m["started_at"] is not None
