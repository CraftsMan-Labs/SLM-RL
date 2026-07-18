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
import yaml

from slm_rl.config.loader import load_game_config
from slm_rl.playground import experiments as exp_mod
from slm_rl.playground.experiments import (
    Busy,
    ExperimentDir,
    InvalidExperiment,
    _materialized_dqn_checkpoint,
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
    for kind in ("quick", "evolve", "theater", "bake"):
        exp_mod._ACTIVE[kind] = None
        exp_mod._ACTIVE_OWNER[kind] = None
    yield
    for kind in ("quick", "evolve", "theater", "bake"):
        exp_mod._ACTIVE[kind] = None
        exp_mod._ACTIVE_OWNER[kind] = None


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


def test_knobs_schema_scoped_per_game():
    """Irrelevant game.extra knobs must not appear (None default = drop)."""
    bx = {k["key"]: k for k in knobs_schema("boxing")}
    fw = {k["key"]: k for k in knobs_schema("freeway")}
    assert "score_scale" in bx
    assert "score_scale" in fw
    # shared run knobs always present
    assert "episodes_per_generation" in bx
    assert "episodes_per_generation" in fw
    assert all(k["default"] is not None for k in bx.values())
    assert all("target" in k for k in bx.values())
    assert all("help" in k and k["help"]["body"] for k in bx.values())


def test_knobs_schema_help_cites_game_default():
    """Hover copy must name this game's recommended (pre-filled) default."""
    da = {k["key"]: k for k in knobs_schema("demon-attack")}
    art = da["action_repeat_threshold"]
    assert art["default"] == 300
    assert "Recommended default for demon-attack: 300" in art["help"]["body"]
    si = {k["key"]: k for k in knobs_schema("space-invaders")}
    assert "Recommended default for space-invaders: 88" in (
        si["action_repeat_threshold"]["help"]["body"]
    )


def test_create_experiment_dqn_uses_pack_checkpoint(tmp_path: Path):
    """teacher=dqn must resolve packs/<game>/dqn.pt, not space-invaders."""
    pack_pt = tmp_path / "packs" / "demon-attack" / "dqn.pt"
    pack_pt.parent.mkdir(parents=True)
    pack_pt.write_bytes(b"fake-ckpt")
    exp = create_experiment(
        tmp_path,
        "demon-attack",
        "dqn-pack",
        knob_values={"teacher": "dqn"},
    )
    data = yaml.safe_load((exp.config_dir / "default.yaml").read_text(encoding="utf-8"))
    assert data["teacher"]["dqn_checkpoint"] == str(pack_pt.resolve())


def test_create_experiment_dqn_missing_checkpoint_fails(tmp_path: Path):
    with pytest.raises(InvalidExperiment, match="Paste a Hugging Face repo"):
        create_experiment(
            tmp_path,
            "boxing",
            "no-ckpt",
            knob_values={"teacher": "dqn"},
        )


def test_launch_bake_appends_log_instead_of_truncating(tmp_path: Path, monkeypatch):
    from slm_rl.playground.experiments import bake_log_path, launch_bake, tail_bake_log

    log = bake_log_path(tmp_path)
    log.parent.mkdir(parents=True)
    log.write_text("older bake line\n", encoding="utf-8")

    monkeypatch.setattr(exp_mod, "_spawn", lambda cmd, log_path, env=None, append=False: _FakePopen())
    launch_bake(tmp_path, game="boxing", episodes=10, dqn_decisions=0)
    text = log.read_text(encoding="utf-8")
    assert "older bake line" in text
    assert "========== bake" in text
    assert "game=boxing" in text
    assert "older bake line" in tail_bake_log(tmp_path)


def test_create_experiment_dqn_downloads_from_hf_url(tmp_path: Path, monkeypatch):
    fake_pt = tmp_path / "packs" / "org__boxing__dqn" / "dqn.pt"
    fake_pt.parent.mkdir(parents=True)
    fake_pt.write_bytes(b"ckpt")

    def fake_resolve(url, home, game):
        assert url == "org/slm-rl-boxing"
        assert game == "boxing"
        return fake_pt

    monkeypatch.setattr("slm_rl.packs.resolve_dqn", fake_resolve)
    exp = create_experiment(
        tmp_path,
        "boxing",
        "dqn-hf",
        knob_values={"teacher": "dqn"},
        dqn_url="https://huggingface.co/datasets/org/slm-rl-boxing",
    )
    data = yaml.safe_load((exp.config_dir / "default.yaml").read_text(encoding="utf-8"))
    assert data["teacher"]["dqn_checkpoint"] == str(fake_pt.resolve())
    assert data["dqn_url"] == "org/slm-rl-boxing"
    meta = json.loads(exp.experiment_json.read_text(encoding="utf-8"))
    assert meta["dqn_url"] == "org/slm-rl-boxing"


def test_quick_run_resolves_hf_dqn_even_when_teacher_is_heuristic(
    tmp_path: Path, monkeypatch,
):
    """Quick Run must use configured dqn_url — never train a fresh DQN, and
    must not silently fall back to the heuristic when Hub weights exist."""
    fake_pt = tmp_path / "packs" / "org__boxing__dqn" / "dqn.pt"
    fake_pt.parent.mkdir(parents=True)
    fake_pt.write_bytes(b"ckpt")

    monkeypatch.setattr(
        "slm_rl.packs.resolve_dqn",
        lambda url, home, game: fake_pt,
    )
    # teacher=heuristic leaves dqn_checkpoint null, but dqn_url is still stored
    # (workshop create form often picks heuristic while pasting a HF DQN URL).
    exp = create_experiment(
        tmp_path,
        "boxing",
        "quick-hf",
        knob_values={"teacher": "heuristic"},
        dqn_url="org/slm-rl-boxing",
    )
    data = yaml.safe_load((exp.config_dir / "default.yaml").read_text(encoding="utf-8"))
    assert data.get("teacher", {}).get("dqn_checkpoint") in (None, "")
    assert data["dqn_url"] == "org/slm-rl-boxing"

    path = _materialized_dqn_checkpoint(exp)
    assert path == str(fake_pt.resolve())

    captured: list[list[str]] = []

    def fake_spawn(cmd, log_path, env=None, append=False):
        captured.append(cmd)
        return _FakePopen()

    monkeypatch.setattr(exp_mod, "_spawn", fake_spawn)
    launch_rollout(exp, "boxing", agent="solver", episodes=1, seed=0)
    assert "--dqn-checkpoint" in captured[0]
    assert str(fake_pt.resolve()) in captured[0]


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


def test_experiment_stats_tail_scan_on_huge_jsonl(tmp_path: Path, monkeypatch):
    """Oversized Atari JSONL must not full-scan — only the trailing window."""
    import slm_rl.playground.stats as stats_mod

    monkeypatch.setattr(stats_mod, "_MAX_BYTES_PER_FILE", 800)
    stats_mod._CACHE.clear()

    run_dir = tmp_path / "pg-huge"
    path = run_dir / "generations" / "gen_001" / "rollouts" / "game.jsonl"
    path.parent.mkdir(parents=True)

    # Pad so the first terminal episode falls outside the tail window; the
    # last episode's score:999 must still be visible.
    pad = json.dumps(_rec(episode_id="old", step_idx=0, parsed_action="NOOP")) + "\n"
    pad *= 40  # well over 800 bytes
    tail = (
        json.dumps(_rec(episode_id="new", step_idx=0, parsed_action="FIRE")) + "\n"
        + json.dumps(
            _rec(
                episode_id="new", step_idx=1, parsed_action="FIRE",
                outcome="score:999", terminated=True,
            )
        )
        + "\n"
    )
    path.write_text(pad + tail, encoding="utf-8")
    assert path.stat().st_size > 800

    stats = experiment_stats(run_dir)
    assert stats["max_score"] == 999.0
    assert stats["mean_score"] == 999.0
    assert stats["episodes"] >= 1


# --- Test 7: busy lock ------------------------------------------------------


class _FakePopen:
    """Looks alive to _busy() (poll() returns None) until terminated."""

    _next_pid = 10_000

    def __init__(self):
        self.pid = _FakePopen._next_pid
        _FakePopen._next_pid += 1
        self._alive = True

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self._alive = False

    def kill(self):
        self._alive = False

    def wait(self, timeout=None):
        self._alive = False
        return 0


def test_second_launch_rollout_while_busy_raises(tmp_path: Path, monkeypatch):
    exp = create_experiment(tmp_path, "space-invaders", "busy-test", knob_values={})

    def fake_spawn(cmd, log_path, env=None, append=False):
        return _FakePopen()

    monkeypatch.setattr(exp_mod, "_spawn", fake_spawn)

    launch_rollout(exp, "space-invaders", agent="random", episodes=1, seed=0)
    with pytest.raises(Busy):
        launch_rollout(exp, "space-invaders", agent="random", episodes=1, seed=0)


def test_stop_experiment_terminates_owned_job(tmp_path: Path, monkeypatch):
    from slm_rl.playground.experiments import NotRunning, stop_experiment

    exp = create_experiment(tmp_path, "space-invaders", "stop-me", knob_values={})
    monkeypatch.setattr(exp_mod, "_spawn", lambda cmd, log_path, env=None, append=False: _FakePopen())
    monkeypatch.setattr(exp_mod, "_STOP_GRACE_SECONDS", 0.05)

    launch_rollout(exp, "space-invaders", agent="random", episodes=1, seed=0)
    assert exp_mod.active_jobs_for("stop-me") == ["quick"]
    stopped = stop_experiment("stop-me")
    assert stopped == ["quick"]
    assert exp_mod.active_jobs_for("stop-me") == []
    with pytest.raises(NotRunning):
        stop_experiment("stop-me")


def test_stop_experiment_ignores_other_owners(tmp_path: Path, monkeypatch):
    from slm_rl.playground.experiments import NotRunning, launch_evolve, stop_experiment

    a = create_experiment(tmp_path, "space-invaders", "alpha", knob_values={})
    b = create_experiment(tmp_path, "space-invaders", "beta", knob_values={})
    monkeypatch.setattr(exp_mod, "_spawn", lambda cmd, log_path, env=None, append=False: _FakePopen())
    launch_evolve(a, "space-invaders", generations=1)
    with pytest.raises(NotRunning):
        stop_experiment("beta")
    assert exp_mod.active_jobs_for("alpha") == ["evolve"]
    assert b.name == "beta"


def test_launch_evolve_forwards_profile_hf_token(tmp_path: Path, monkeypatch):
    """Welcome-screen token must reach the evolve child as HF_TOKEN."""
    from slm_rl.playground.experiments import launch_evolve, stop_experiment

    exp = create_experiment(tmp_path, "boxing", "token-fwd", knob_values={})
    seen: dict[str, object] = {}

    def fake_spawn(cmd, log_path, env=None, append=False):
        seen["env"] = env
        return _FakePopen()

    monkeypatch.setattr(exp_mod, "_spawn", fake_spawn)
    monkeypatch.setattr(exp_mod, "_STOP_GRACE_SECONDS", 0.05)
    launch_evolve(exp, "boxing", generations=1, token="hf_testtoken1234")
    env = seen["env"]
    assert isinstance(env, dict)
    assert env["HF_TOKEN"] == "hf_testtoken1234"
    assert env["HUGGING_FACE_HUB_TOKEN"] == "hf_testtoken1234"
    stop_experiment("token-fwd")


def test_stop_experiment_can_target_theater_only(tmp_path: Path, monkeypatch):
    from slm_rl.playground.experiments import launch_evolve, launch_theater, stop_experiment

    exp = create_experiment(tmp_path, "space-invaders", "keep-train", knob_values={})
    monkeypatch.setattr(exp_mod, "_spawn", lambda cmd, log_path, env=None, append=False: _FakePopen())
    monkeypatch.setattr(exp_mod, "_STOP_GRACE_SECONDS", 0.05)
    launch_evolve(exp, "space-invaders", generations=1)
    launch_theater(exp, "space-invaders", episodes=2)
    assert set(exp_mod.active_jobs_for("keep-train")) == {"evolve", "theater"}
    stopped = stop_experiment("keep-train", kinds=["theater"])
    assert stopped == ["theater"]
    assert exp_mod.active_jobs_for("keep-train") == ["evolve"]


# --- Test 8: HTTP smoke ------------------------------------------------------


def test_http_create_experiment_busy_and_knobs(tmp_path: Path, monkeypatch):
    from slm_rl.playground import server as server_mod

    # Avoid spawning a real subprocess in this HTTP test: fake_spawn looks
    # alive forever, which is exactly what we need to exercise the 409 path.
    monkeypatch.setattr(exp_mod, "_spawn", lambda cmd, log_path, env=None, append=False: _FakePopen())

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
        busy_body = json.loads(resp.read().decode("utf-8"))
        conn.close()
        assert "Busy" in busy_body["error"]

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", "/api/experiments/http-test/stop", body=b"{}",
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        stop_body = json.loads(resp.read().decode("utf-8"))
        conn.close()
        assert resp.status == 200
        assert stop_body["stopped"] == ["quick"]

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/api/experiments")
        resp = conn.getresponse()
        rows = json.loads(resp.read().decode("utf-8"))
        conn.close()
        assert resp.status == 200
        http_row = next(r for r in rows if r["name"] == "http-test")
        assert http_row["active_jobs"] == []
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5.0)


# --- Plan 026 Phase C: multi-game playground ---------------------------------


def test_create_experiment_persists_game(tmp_path: Path):
    exp = create_experiment(tmp_path, "boxing", "ms-exp", knob_values={})
    data = json.loads(exp.experiment_json.read_text(encoding="utf-8"))
    assert data["game"] == "boxing"
    raw = (exp.config_dir / "default.yaml").read_text(encoding="utf-8")
    assert "game: boxing" in raw or 'game: "boxing"' in raw or "game: boxing\n" in raw


def test_http_games_list_and_knobs_per_game(tmp_path: Path, monkeypatch):
    from http.server import ThreadingHTTPServer

    from slm_rl.games.registry import available_games
    from slm_rl.playground import server as server_mod

    monkeypatch.setattr(exp_mod, "_spawn", lambda cmd, log_path, env=None, append=False: _FakeAlwaysDone())

    handler_cls = server_mod._make_handler(tmp_path, "boxing")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/api/games")
        resp = conn.getresponse()
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))
        conn.close()
        assert payload["default"] == "boxing"
        assert "boxing" in payload["games"]
        assert set(payload["games"]) == set(available_games())

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/api/knobs?game=boxing")
        resp = conn.getresponse()
        assert resp.status == 200
        knobs = json.loads(resp.read().decode("utf-8"))
        conn.close()
        assert len(knobs) == len(knobs_schema("boxing"))
        keys = {k["key"] for k in knobs}
        assert "score_scale" in keys
        assert "width" not in keys
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5.0)


def test_http_multi_game_create_and_list_filter(tmp_path: Path, monkeypatch):
    """One process, two games: create both, scoreboard rows carry game."""
    from http.server import ThreadingHTTPServer

    from slm_rl.playground import server as server_mod

    monkeypatch.setattr(exp_mod, "_spawn", lambda cmd, log_path, env=None, append=False: _FakeAlwaysDone())

    # Default form preselect is space-invaders; body overrides per create.
    handler_cls = server_mod._make_handler(tmp_path, "space-invaders")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    try:
        for name, game in (("exp-bx", "boxing"), ("exp-fw", "freeway")):
            payload = json.dumps({
                "name": name,
                "game": game,
                "knob_values": {},
                "agent": "random",
                "episodes": 1,
            }).encode("utf-8")
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request(
                "POST", "/api/experiments", body=payload,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            body = json.loads(resp.read().decode("utf-8"))
            conn.close()
            assert resp.status == 200, body
            assert body["game"] == game
            meta = json.loads(
                (tmp_path / "playground" / name / "experiment.json").read_text(encoding="utf-8")
            )
            assert meta["game"] == game

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/api/experiments")
        resp = conn.getresponse()
        assert resp.status == 200
        rows = json.loads(resp.read().decode("utf-8"))
        conn.close()

        by_name = {r["name"]: r for r in rows}
        assert by_name["exp-bx"]["game"] == "boxing"
        assert by_name["exp-fw"]["game"] == "freeway"
        assert by_name["baseline"]["game"] == "space-invaders"
        assert isinstance(by_name["exp-bx"]["knob_values"], dict)
        assert isinstance(by_name["exp-fw"]["knob_values"], dict)
        # Client-side filter chips key off distinct game values in the list.
        games_present = {r["game"] for r in rows if r.get("game")}
        assert "boxing" in games_present
        assert "freeway" in games_present
        boxing_only = [r for r in rows if r.get("game") == "boxing"]
        assert any(r["name"] == "exp-bx" for r in boxing_only)
        assert all(r["name"] != "exp-fw" for r in boxing_only)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5.0)


def test_page_has_game_select_and_filter_chips():
    from slm_rl.playground.page import PAGE

    assert 'id="f-game"' in PAGE
    assert 'id="game-filters"' in PAGE
    assert "/api/games" in PAGE
    assert "/api/knobs?game=" in PAGE


# --- Plan 026 Phase F: live log tailer --------------------------------------


def test_tail_log_returns_last_bytes(tmp_path: Path):
    from slm_rl.playground.experiments import LOG_TAIL_BYTES, tail_log

    exp = create_experiment(tmp_path, "boxing", "log-exp", knob_values={})
    assert tail_log(exp, "rollout") == ""  # missing file → empty

    path = exp.log_path("rollout")
    path.write_text("hello\nworld\n", encoding="utf-8")
    assert tail_log(exp, "rollout") == "hello\nworld\n"

    # Cap: content longer than LOG_TAIL_BYTES returns only the tail.
    big = ("X" * 100) + ("Y" * LOG_TAIL_BYTES)
    path.write_text(big, encoding="utf-8")
    got = tail_log(exp, "rollout")
    assert len(got.encode("utf-8")) == LOG_TAIL_BYTES
    assert got.endswith("Y" * 100)
    assert not got.startswith("X")


def test_http_log_tail_kind_validation_and_404(tmp_path: Path, monkeypatch):
    from http.server import ThreadingHTTPServer

    from slm_rl.playground import server as server_mod

    monkeypatch.setattr(exp_mod, "_spawn", lambda cmd, log_path, env=None, append=False: _FakeAlwaysDone())

    exp = create_experiment(tmp_path, "boxing", "log-http", knob_values={})
    exp.log_path("evolve").write_text("gen 0 start\ngen 0 done\n", encoding="utf-8")

    handler_cls = server_mod._make_handler(tmp_path, "boxing")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/api/experiments/log-http/log?kind=evolve")
        resp = conn.getresponse()
        body = resp.read().decode("utf-8")
        conn.close()
        assert resp.status == 200
        assert resp.getheader("Content-Type", "").startswith("text/plain")
        assert body == "gen 0 start\ngen 0 done\n"

        # Known exp, missing log file → 200 empty body (not 404).
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/api/experiments/log-http/log?kind=theater")
        resp = conn.getresponse()
        empty = resp.read()
        conn.close()
        assert resp.status == 200
        assert empty == b""

        # Bad kind → 400 JSON.
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/api/experiments/log-http/log?kind=nope")
        resp = conn.getresponse()
        err = json.loads(resp.read().decode("utf-8"))
        conn.close()
        assert resp.status == 400
        assert "kind" in err["error"]

        # Unknown experiment name → 404.
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/api/experiments/does-not-exist/log?kind=rollout")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        assert resp.status == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5.0)


def test_page_has_log_panel():
    from slm_rl.playground.page import PAGE

    assert 'id="log-panel"' in PAGE
    assert 'id="log-body"' in PAGE
    assert "/log?kind=" in PAGE
    assert "openLog" in PAGE


class _FakeAlwaysDone:
    """poll() returns non-None immediately — looks finished, never busy."""

    _next_pid = 20_000

    def __init__(self):
        self.pid = _FakeAlwaysDone._next_pid
        _FakeAlwaysDone._next_pid += 1

    def poll(self):
        return 0


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
