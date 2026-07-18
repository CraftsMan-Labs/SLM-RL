"""DQN Teachers monitor: log parse + job discovery (no torch)."""

from __future__ import annotations

from pathlib import Path

from slm_rl.playground.dqn_monitor import (
    job_metrics,
    list_dqn_jobs,
    parse_train_log_metrics,
)


SAMPLE = """
A.L.E: Arcade Learning Environment
decisions=500 episodes=1 eps=0.999 mean_ep_reward_last20=0.1000 loss=n/a
decisions=1000 episodes=2 eps=0.998 mean_ep_reward_last20=0.2000 loss=0.0123
eval decisions=1000 episodes=5 mean_ep_reward=0.4500
"""


def test_parse_train_and_eval_lines():
    rows = parse_train_log_metrics(SAMPLE)
    assert [r["split"] for r in rows] == ["train", "train", "eval"]
    assert rows[1]["loss"] == 0.0123
    assert rows[0]["loss"] is None
    assert rows[2]["mean_ep_reward"] == 0.45


def test_list_jobs_from_log_and_checkpoint(tmp_path: Path):
    home = tmp_path / "runs"
    teachers = home / "teachers"
    teachers.mkdir(parents=True)
    (teachers / "dqn-boxing.pt").write_bytes(b"fake")
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "train-dqn-boxing.log").write_text(SAMPLE, encoding="utf-8")
    (logs / "train-dqn-boxing.pid").write_text("1\n", encoding="utf-8")

    jobs = list_dqn_jobs(home)
    assert len(jobs) == 1
    assert jobs[0]["game"] == "boxing"
    assert jobs[0]["log_path"]
    assert jobs[0]["last_decisions"] == 1000
    assert jobs[0]["last_eval_reward"] == 0.45

    m = job_metrics(home, "boxing")
    assert len(m["train"]) == 2
    assert len(m["eval"]) == 1
