"""Workshop playground (plan 013): experiment materialization, stats over
synthetic JSONL, busy-lock behavior, and an HTTP smoke test. Stdlib + pytest
only for tests 4-8. Test 9 (end-to-end with a real subprocess) needs the
[atari] extra and is marked accordingly; it uses --agent random only, never
an LLM (CODING_GUIDELINE: never load a model in tests)."""

from __future__ import annotations

import http.client
import json
import subprocess
import threading
import time
from pathlib import Path

import pytest

from slm_rl.config.loader import load_game_config
from slm_rl.playground import experiments as exp_mod
from slm_rl.playground.experiments import (
    Busy,
    ExperimentDir,
    InvalidExperiment,
    create_experiment,
    launch_rollout,
    validate_name,
)
from slm_rl.playground.knobs import knobs_schema
from slm_rl.playground.stats import experiment_stats


@pytest.fixture(autouse=True)
def _reset_locks():
    """Each test gets a clean subprocess-lock state; the module-level locks
    are process-global (by design -- one quick + one evolve run at a time
    across the whole server), so tests must not leak busy state into each
    other."""
    for kind in ("quick", "evolve"):
        exp_mod._ACTIVE[kind] = None
    yield
    for kind in ("quick", "evolve"):
        exp_mod._ACTIVE[kind] = None


# --- Test 4: create_experiment materializes + round-trips knob values -----


def test_create_experiment_materializes_and_round_trips(tmp_path: Path):
    exp = create_experiment(
        tmp_path,
        "space-invaders",
        "my-experiment",
        knob_values={"max_turns": 123, "score_scale": 15.0, "teacher": "heuristic"},
    )
    assert exp.config_dir.exists()
    game_cfg = load_game_config("space-invaders", config_dir=exp.config_dir)
    assert game_cfg.max_turns == 123
    assert game_cfg.extra["score_scale"] == 15.0


def test_create_experiment_rejects_bad_name(tmp_path: Path):
    with pytest.raises(InvalidExperiment):
        create_experiment(tmp_path, "space-invaders", "../evil", knob_values={})


def test_validate_name_accepts_kebab_case():
    validate_name("tighter-loop-2")  # must not raise


# --- Test 5: reward code with a syntax error is rejected, nothing written --


def test_reward_code_syntax_error_rejected_without_writing(tmp_path: Path):
    with pytest.raises(InvalidExperiment):
        create_experiment(
            tmp_path,
            "space-invaders",
            "bad-code",
            knob_values={},
            reward_code="def shape_reward(ctx)\n    return 1\n",  # missing colon
        )
    assert not (tmp_path / "playground" / "bad-code").exists()


def test_valid_reward_code_is_written_and_wired_into_extra(tmp_path: Path):
    exp = create_experiment(
        tmp_path,
        "space-invaders",
        "good-code",
        knob_values={},
        reward_code="def shape_reward(ctx):\n    return ctx['default_reward']\n",
    )
    assert exp.reward_hook_path.exists()
    game_cfg = load_game_config("space-invaders", config_dir=exp.config_dir)
    assert game_cfg.extra["reward_hook"] == str(exp.reward_hook_path.resolve())


# --- Test 6: experiment_stats over a hand-written synthetic JSONL ----------


def _rec(**fields):
    base = {
        "run_id": "pg-x", "generation": 0, "game": "space-invaders",
        "episode_id": "ep1", "step_idx": 0, "seed": 0, "model_id": "m",
        "adapter_ref": None, "opponent_id": None, "prompt_messages": [],
        "completion": "", "parsed_action": "FIRE", "legal_actions": ["FIRE"],
        "parse_status": "ok", "reward": 0.1, "shaped_reward": 0.1,
        "cum_reward": 0.1, "terminated": False, "truncated": False,
        "outcome": None, "state_hash": "abc", "monitor_flags": {}, "timestamp": "",
    }
    base.update(fields)
    return base


def test_experiment_stats_over_synthetic_two_file_jsonl(tmp_path: Path):
    run_dir = tmp_path / "pg-x"
    f1 = run_dir / "generations" / "gen_000" / "rollouts" / "a.jsonl"
    f2 = run_dir / "generations" / "gen_000" / "rollouts" / "b.jsonl"
    f1.parent.mkdir(parents=True)

    # episode 1: 2 decisions, ends with score 100, one has monitor_flags.
    with f1.open("w", encoding="utf-8") as f:
        f.write(json.dumps(_rec(episode_id="ep1", step_idx=0, parsed_action="FIRE",
                                 monitor_flags={"intervention": {"kind": "reflect"}})) + "\n")
        f.write(json.dumps(_rec(episode_id="ep1", step_idx=1, parsed_action="RIGHT",
                                 outcome="score:100", terminated=True)) + "\n")
    # episode 2 and episode 3 interleaved in the second file.
    with f2.open("w", encoding="utf-8") as f:
        f.write(json.dumps(_rec(episode_id="ep2", step_idx=0, parsed_action="FIRE")) + "\n")
        f.write(json.dumps(_rec(episode_id="ep3", step_idx=0, parsed_action="LEFT")) + "\n")
        f.write(json.dumps(_rec(episode_id="ep2", step_idx=1, parsed_action="FIRE",
                                 outcome="score:200", terminated=True)) + "\n")
        f.write(json.dumps(_rec(episode_id="ep3", step_idx=1, parsed_action="FIRE",
                                 outcome="score:300", terminated=True)) + "\n")

    stats = experiment_stats(run_dir)

    assert stats["episodes"] == 3
    assert stats["mean_score"] == 200.0
    assert stats["median_score"] == 200.0
    assert stats["max_score"] == 300.0
    assert stats["intervention_episodes"] == 1
    assert stats["status"] == "complete"
    assert sum(stats["action_mix"].values()) == pytest.approx(100.0, abs=0.5)


def test_experiment_stats_no_data(tmp_path: Path):
    stats = experiment_stats(tmp_path / "does-not-exist")
    assert stats["episodes"] == 0
    assert stats["status"] == "no_data"
    assert stats["mean_score"] is None


# --- Test 7: busy lock ------------------------------------------------------


class _FakePopen:
    """Looks alive to _busy() (poll() returns None) until killed."""

    def poll(self):
        return None


def test_second_launch_rollout_while_busy_raises(tmp_path: Path, monkeypatch):
    exp = create_experiment(tmp_path, "space-invaders", "busy-test", knob_values={})

    def fake_spawn(cmd, log_path):
        return _FakePopen()

    monkeypatch.setattr(exp_mod, "_spawn", fake_spawn)

    launch_rollout(exp, "space-invaders", agent="random", episodes=1, seed=0)
    with pytest.raises(Busy):
        launch_rollout(exp, "space-invaders", agent="random", episodes=1, seed=0)


# --- Test 8: HTTP smoke ------------------------------------------------------


def test_http_create_experiment_busy_and_knobs(tmp_path: Path, monkeypatch):
    from slm_rl.playground import server as server_mod

    # Avoid spawning a real subprocess in this HTTP test: fake_spawn looks
    # alive forever, which is exactly what we need to exercise the 409 path.
    monkeypatch.setattr(exp_mod, "_spawn", lambda cmd, log_path: _FakePopen())

    handler_cls = server_mod._make_handler(tmp_path, "space-invaders")
    from http.server import ThreadingHTTPServer

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/api/knobs")
        resp = conn.getresponse()
        assert resp.status == 200
        knobs = json.loads(resp.read().decode("utf-8"))
        conn.close()
        assert len(knobs) == len(knobs_schema("space-invaders"))
        for k in knobs:
            assert "default" in k

        payload = json.dumps(
            {"name": "http-test", "knob_values": {}, "agent": "random", "episodes": 1}
        ).encode("utf-8")
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", "/api/experiments", body=payload,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        assert resp.status == 200
        resp.read()
        conn.close()
        assert (tmp_path / "playground" / "http-test").exists()

        # Second POST while the fake subprocess is "alive" -> 409.
        payload2 = json.dumps(
            {"name": "http-test-2", "knob_values": {}, "agent": "random", "episodes": 1}
        ).encode("utf-8")
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", "/api/experiments", body=payload2,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        assert resp.status == 409
        resp.read()
        conn.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5.0)


# --- Test 9: end-to-end with a real subprocess (needs [atari]) ------------


def test_end_to_end_quick_experiment_real_subprocess(tmp_path: Path):
    pytest.importorskip("ale_py")
    pytest.importorskip("gymnasium")

    exp = create_experiment(
        tmp_path,
        "space-invaders",
        "e2e",
        knob_values={"max_turns": 40},
    )
    proc = launch_rollout(exp, "space-invaders", agent="random", episodes=2, seed=0)

    deadline = time.monotonic() + 120.0
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.5)
    assert proc.poll() is not None, "rollout subprocess did not finish within 120s"
    assert proc.poll() == 0, exp.log_path("rollout").read_text(encoding="utf-8")

    stats = experiment_stats(exp.run_dir)
    assert stats["episodes"] == 2
    assert stats["mean_score"] is not None
    assert isinstance(stats["mean_score"], float)
