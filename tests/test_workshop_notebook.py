"""CPU-safe checks for the Colab workshop notebook and lab helpers."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKSHOP = ROOT / "docs" / "workshop"
sys.path.insert(0, str(WORKSHOP))

from lab import (  # noqa: E402
    ask,
    bind_tracker,
    clamp_float,
    clamp_int,
    complete_chapter,
    connect_workshop_tracker,
    ensure_card,
    get_tracker,
    grade,
    load_wst_api_key,
    new_card,
    pick_action,
    record_guess,
    require_names,
    resolve_backend,
    resolve_game,
    resolve_mode,
    sanitize_run_name,
    scorecard,
    show_card,
    skip_gates_enabled,
    start_chapter,
)


class _Act:
    def __init__(self, id: str, label: str) -> None:
        self.id = id
        self.label = label


def test_lab_clamps_and_sanitizes():
    assert clamp_int(99, 0, 8, "n") == 8
    assert clamp_int("nope", 1, 4, "n") == 1
    assert clamp_float(2.5, 0.0, 1.5, "t") == 1.5
    assert sanitize_run_name("my run!!") == "my-run"
    assert sanitize_run_name("///") == "colab"
    assert resolve_game("freeway") == "freeway"
    assert resolve_game("pong") == "boxing"
    assert resolve_mode("full") == "FULL"
    assert resolve_backend("q4") == "transformers-4bit"
    assert resolve_backend("auto") is None


def test_lab_pick_action_and_grade():
    legal = [_Act("FIRE", "punch"), _Act("LEFT", "move left")]
    assert pick_action(legal, "FIRE").id == "FIRE"
    assert pick_action(legal, "move left").id == "LEFT"
    assert pick_action(legal, "2").id == "LEFT"
    assert pick_action(legal, "NOPE").id == "FIRE"
    assert "correct" in grade("ok", "ok")
    assert "result was" in grade("ok", "fallback_random")
    assert grade("not sure", "ok").startswith("result:")


def test_lab_require_and_scorecard(capsys):
    require_names({"a": 1, "b": 2}, "a", "b")
    with pytest.raises(RuntimeError, match="Missing: c"):
        require_names({"a": 1}, "a", "c")
    scorecard("demo", [("k", 1)])
    out = capsys.readouterr().out
    assert "=== demo ===" in out
    assert "k" in out


def test_ask_skip_and_reader(monkeypatch):
    assert ask("name", default="Ada", skip=True) == "Ada"
    assert ask("pick", allowed=("ok", "retry_ok"), default="", skip=True) == "ok"
    assert ask("pick", allowed=("ok", "retry_ok"), default="RETRY_OK", skip=True) == "retry_ok"
    answers = iter(["nope", "OK"])
    assert (
        ask(
            "parse?",
            allowed=("ok", "retry_ok"),
            default="ok",
            skip=False,
            reader=lambda _prompt: next(answers),
        )
        == "ok"
    )
    names = iter(["  ", "Ada"])
    assert ask("name", default="anon", skip=False, reader=lambda _prompt: next(names)) == "Ada"
    monkeypatch.setenv("WORKSHOP_SKIP_GATES", "1")
    assert skip_gates_enabled(False) is True
    assert ask("name", default="env", skip=None) == "env"
    monkeypatch.delenv("WORKSHOP_SKIP_GATES", raising=False)
    assert skip_gates_enabled(False) is False


def test_ask_hf_token_allows_blank_and_skips_when_seeded(monkeypatch):
    from lab import ask_hf_token

    assert ask_hf_token(default="hf_seededtoken", skip=True) == "hf_seededtoken"
    assert ask_hf_token(default="", skip=True) == ""
    assert ask_hf_token(default="hf_from_form", reader=lambda _prompt: "should-not-run") == (
        "hf_from_form"
    )
    assert ask_hf_token(default="", reader=lambda _prompt: "  hf_pasted  ") == "hf_pasted"
    assert ask_hf_token(default="", reader=lambda _prompt: "   ") == ""
    monkeypatch.setenv("WORKSHOP_SKIP_GATES", "1")
    assert ask_hf_token(default="hf_envskip") == "hf_envskip"
    monkeypatch.delenv("WORKSHOP_SKIP_GATES", raising=False)


def test_card_records_and_prints(capsys):
    card = new_card(" Ada ")
    assert card["name"] == "Ada"
    record_guess(card, "parse", "ok", "ok")
    record_guess(card, "gate", "promote", "reject")
    record_guess(card, "skip", "not sure", "ok")
    show_card(card)
    out = capsys.readouterr().out
    assert "Ada's scorecard" in out
    assert "1/2 correct" in out
    ns: dict = {}
    created = ensure_card(ns)
    assert created["name"] == "anonymous"
    assert ns["CARD"] is created


class _FakeTracker:
    def __init__(self, me: dict | None = None, fail_status: int | None = None) -> None:
        self.me_payload = me if me is not None else {"run": {"name": "demo"}, "progress": []}
        self.fail_status = fail_status
        self.calls: list[tuple[str, str]] = []

    def me(self) -> dict:
        return self.me_payload

    def start(self, key: str) -> dict:
        self.calls.append(("start", key))
        if self.fail_status is not None:
            raise _FakeHTTPError(self.fail_status)
        return {"key": key, "status": "in_progress"}

    def complete(self, key: str) -> dict:
        self.calls.append(("complete", key))
        if self.fail_status is not None:
            raise _FakeHTTPError(self.fail_status)
        return {"key": key, "status": "completed"}


class _FakeHTTPError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.response = type("R", (), {"status_code": status_code})()


def test_tracker_bind_start_complete(capsys, monkeypatch):
    import lab as lab_mod

    monkeypatch.setattr(lab_mod, "_TRACKER", None)
    fake = _FakeTracker()
    bind_tracker(fake)
    assert get_tracker() is fake
    start_chapter(1)
    complete_chapter(1)
    assert fake.calls == [("start", "chapter-1"), ("complete", "chapter-1")]
    out = capsys.readouterr().out
    assert "chapter-1 → in_progress" in out
    assert "chapter-1 → completed" in out


def test_tracker_held_chapter_raises(monkeypatch):
    import lab as lab_mod

    monkeypatch.setattr(lab_mod, "_TRACKER", None)
    bind_tracker(_FakeTracker(fail_status=403))
    with pytest.raises(RuntimeError, match="held"):
        start_chapter(2)


def test_connect_workshop_tracker_requires_key(monkeypatch):
    import lab as lab_mod

    monkeypatch.setattr(lab_mod, "_TRACKER", None)
    monkeypatch.delenv("WST_API_KEY", raising=False)
    monkeypatch.delenv("WORKSHOP_SKIP_GATES", raising=False)
    monkeypatch.setattr(lab_mod, "load_wst_api_key", lambda: "")
    with pytest.raises(RuntimeError, match="Missing participant API key"):
        connect_workshop_tracker(join_url="https://workshop.craftsmanlabs.net/join/demo")


def test_connect_workshop_tracker_skips_without_key(monkeypatch, capsys):
    import lab as lab_mod

    monkeypatch.setattr(lab_mod, "_TRACKER", None)
    monkeypatch.setenv("WORKSHOP_SKIP_GATES", "1")
    monkeypatch.setattr(lab_mod, "load_wst_api_key", lambda: "")
    assert connect_workshop_tracker(require=True) is None
    assert "progress skipped" in capsys.readouterr().out
    monkeypatch.delenv("WORKSHOP_SKIP_GATES", raising=False)


def test_load_wst_api_key_from_env(monkeypatch):
    monkeypatch.setenv("WST_API_KEY", " wsp_live_test ")
    assert load_wst_api_key() == "wsp_live_test"
    monkeypatch.delenv("WST_API_KEY", raising=False)
    assert load_wst_api_key() == ""
    assert load_wst_api_key("  wsp_from_form  ") == "wsp_from_form"


def test_ask_wst_api_key_prefers_seed_and_prompts(monkeypatch):
    from lab import ask_wst_api_key

    assert ask_wst_api_key(default="wsp_live_seeded", skip=True) == "wsp_live_seeded"
    assert ask_wst_api_key(default="", skip=True) == ""
    assert ask_wst_api_key(default="wsp_form", reader=lambda _p: "should-not-run") == "wsp_form"
    assert ask_wst_api_key(default="", reader=lambda _p: "  wsp_pasted  ") == "wsp_pasted"
    monkeypatch.setenv("WORKSHOP_SKIP_GATES", "1")
    assert ask_wst_api_key(default="") == ""
    monkeypatch.delenv("WORKSHOP_SKIP_GATES", raising=False)


def test_committed_notebook_tracks_all_chapters_and_hides_admin_keys():
    nb = json.loads((ROOT / "colab_workshop.ipynb").read_text(encoding="utf-8"))
    joined = "\n".join("".join(c.get("source") or []) for c in nb["cells"])
    assert "WORKSHOP_JOIN_URL" in joined
    assert "https://workshop.craftsmanlabs.net/join/slm-rl-test-run-056892" in joined
    assert "connect_workshop_tracker" in joined
    assert "WST_API_KEY" in joined
    assert "# @title Load WorkShopTracker API key" in joined
    assert "ask_wst_api_key" in joined
    assert "workshop-tracker-client" not in joined
    assert "WorkShopTracker.git" not in joined
    assert "WST_ADMIN_KEY" not in joined
    assert "MASTER_KEY_NOTEBOOK" not in joined
    starts = [joined.find(f"start_chapter({n})") for n in range(14)]
    completes = [joined.find(f"complete_chapter({n})") for n in range(14)]
    assert all(i >= 0 for i in starts)
    assert all(i >= 0 for i in completes)
    assert starts == sorted(starts)
    assert completes == sorted(completes)
    for n in range(14):
        assert starts[n] < completes[n]


def test_inline_workshop_tracker_http_helpers(monkeypatch):
    import io
    import json
    import urllib.error

    from lab import WorkshopTracker, _HTTPStatusError

    calls: list[tuple[str, str]] = []

    class _Resp:
        def __init__(self, payload: dict):
            self._raw = json.dumps(payload).encode()

        def read(self):
            return self._raw

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=0):  # noqa: ARG001
        calls.append((req.get_method(), req.full_url))
        if req.full_url.endswith("/api/v1/me"):
            return _Resp({"run": {"name": "demo"}, "progress": []})
        return _Resp({"key": "chapter-1", "status": "in_progress"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    tracker = WorkshopTracker(base_url="https://example.test", api_key="wsp_live_x")
    assert tracker.me()["run"]["name"] == "demo"
    assert tracker.start("chapter-1")["status"] == "in_progress"
    assert ("GET", "https://example.test/api/v1/me") in calls

    def boom(req, timeout=0):  # noqa: ARG001
        raise urllib.error.HTTPError(
            req.full_url, 403, "Forbidden", hdrs=None, fp=io.BytesIO(b'{"message":"not open yet"}')
        )

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(_HTTPStatusError) as err:
        tracker.complete("chapter-2")
    assert err.value.response.status_code == 403


def test_builder_writes_valid_notebook(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_nb", WORKSHOP / "build_nb.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    dest = tmp_path / "colab_workshop.ipynb"
    spec.loader.exec_module(mod)
    mod.OUT = dest
    mod.cells.clear()
    mod.main()
    nb = json.loads(dest.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert nb["metadata"]["colab"]["gpuType"] == "T4"
    sources = ["".join(c.get("source") or []) for c in nb["cells"]]
    joined = "\n".join(sources)
    for heading in (
        "## 0. Setup",
        "## 1. The games",
        "## 2. Config",
        "## 3. The model plays",
        "## 4. Rollout + dataset",
        "## 5. Teachers / hybrid RL",
        "## 6. Packs",
        "## 7. Training",
        "## 8. Eval and the gate",
        "## 9. The evolve loop",
        "## 10. Theater",
        "## 11. Publish",
        "## 12. Build your own game",
        "## 13. Tests",
    ):
        assert heading in joined
    form_titles = [s for s in sources if s.startswith("# @title")]
    assert len(form_titles) >= 10
    form_cells = [
        c
        for c in nb["cells"]
        if c["cell_type"] == "code"
        and "".join(c.get("source") or []).startswith("# @title")
    ]
    assert form_cells
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells
    assert all(
        cell.get("metadata", {}).get("cellView") == "form"
        for cell in code_cells
    )
    for cell in form_cells:
        src = "".join(cell.get("source") or [])
        assert '{display-mode: "form"}' in src.splitlines()[0]
    assert 'GAME = "boxing"' in joined
    assert "PREDICT_PARSE" in joined
    assert 'SCHEMA_FIELD = "completion"' in joined
    assert 'row.get("completion")' in joined
    assert 'row.get("raw_completion")' not in joined
    assert "TRAIN_STRATEGY" in joined
    assert "PUBLISH = False" in joined
    assert "ensure_game" in joined
    assert "close_backend_if_any" in joined
    assert "mermaid.run()" not in joined
    assert "mermaid.initialize" not in joined
    assert "cdn.jsdelivr.net/npm/mermaid" not in joined
    assert "Presentation: cover → join-lobby" in joined
    assert "**Goal.**" in joined
    assert "PLAY_GAME" in joined
    assert "TRAINING_MODE" in joined
    assert "EVAL_STEPS" in joined
    assert "MARIO_MODEL_REPO" in joined
    assert '"checkout", "-B", BRANCH, "FETCH_HEAD"' in joined
    assert 'sys.modules.pop(stale_module, None)' in joined
    assert "target = reward + γ × best next Q" in joined
    assert "mario-pretrained.mp4" not in joined
    assert "select_episodes" in joined
    assert "docs/workshop/assets/diagrams/evolve-loop.svg" in joined
    assert "docs/workshop/assets/deck/HeroVisual.png" in joined
    assert "max-width:min(100%,480px)" in joined
    assert "max-height:min(28vh,240px)" in joined
    assert "max-width:min(100%,180px)" in joined
    assert "![" not in joined
    assert "docs/workshop/assets/deck/meet-the-teacher.mp4" in joined
    assert "docs/workshop/assets/diagrams/dqn-q-values.svg" in joined
    assert "docs/workshop/assets/diagrams/dqn-loop.svg" in joined
    assert "PREDICT_ACTION" in joined
    assert "NEXT_Q_RIGHT" in joined
    assert "learning should {direction} the prediction" in joined
    assert "EPSILON" in joined
    assert "PREDICT_CLIP" not in joined
    assert "export_sft_dataset" in joined
    assert "SKIP_GATES" not in joined
    assert "skip=SKIP_GATES" not in joined
    assert "# @title Join the room" in joined
    assert "# @title Play one move" not in joined
    assert "YOUR_ACTION" not in joined
    assert "ask(" in joined
    assert "ask_hf_token" in joined
    assert 'HF_TOKEN = ""' in joined
    assert "show_card(CARD)" in joined
    assert "Runtime → Run all" in joined
    assert "WORKSHOP_JOIN_URL" in joined
    assert "https://workshop.craftsmanlabs.net/join/slm-rl-test-run-056892" in joined
    assert "connect_workshop_tracker" in joined
    assert "WST_API_KEY" in joined
    assert "# @title Load WorkShopTracker API key" in joined
    assert "ask_wst_api_key" in joined
    assert "workshop-tracker-client" not in joined
    assert "WorkShopTracker.git" not in joined
    assert "WST_ADMIN_KEY" not in joined
    assert "MASTER_KEY" not in joined
    assert "MASTER_KEY_NOTEBOOK" not in joined
    for n in range(14):
        assert f"start_chapter({n})" in joined
        assert f"complete_chapter({n})" in joined


def test_committed_notebook_matches_builder_and_parses():
    committed = ROOT / "colab_workshop.ipynb"
    assert committed.is_file()
    nb = json.loads(committed.read_text(encoding="utf-8"))
    assert nb["cells"], "notebook is empty"
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source") or [])
        if cell["cell_type"] != "code":
            continue
        if src.startswith("%%html") or src.startswith("!"):
            continue
        cleaned = "\n".join(
            line
            for line in src.splitlines()
            if not line.startswith("%") and not line.startswith("!")
        )
        ast.parse(cleaned, filename=f"cell_{i}")


def test_notebook_diagram_refs_resolve():
    dest = WORKSHOP / "assets" / "diagrams"
    nb = json.loads((ROOT / "colab_workshop.ipynb").read_text(encoding="utf-8"))
    joined = "\n".join("".join(c.get("source") or []) for c in nb["cells"])
    names = [
        "evolve-loop",
        "hardware-tier",
        "games-pipeline",
        "config-merge",
        "parse-action",
        "rollout-dataset",
        "dqn-hybrid",
        "dqn-loop",
        "dqn-q-values",
        "dqn-bellman",
        "dqn-replay",
        "dqn-target",
        "dqn-encoders",
        "dqn-math",
        "trace-to-pair",
        "packs",
        "train-strategies",
        "eval-gate",
        "theater",
        "publish",
        "game-abc",
    ]
    for name in names:
        assert f"{name}.svg" in joined
        assert (dest / f"{name}.svg").is_file()
    from import_deck_assets import DECK_FILES
    from mario_lab import CLIP_FILES

    deck = WORKSHOP / "assets" / "deck"
    for name in DECK_FILES:
        assert name in joined
        assert (deck / name).is_file()
    mario = WORKSHOP / "assets" / "mario"
    for name in CLIP_FILES:
        assert (mario / name).is_file()
        assert (mario / name).stat().st_size > 1000


def test_workshop_diagrams_exist_and_are_svg():
    from diagrams.render import RENDERERS, render_all

    dest = WORKSHOP / "assets" / "diagrams"
    written = render_all(dest)
    names = {path.stem for path in written}
    assert set(RENDERERS) <= names
    for path in written:
        text = path.read_text(encoding="utf-8")
        assert text.startswith("<svg")
        assert 'aria-label=' in text


def test_dqn_loop_diagram_is_guess_loop_not_optimal_policy():
    from diagrams.render import RENDERERS, dqn_loop

    assert "dqn-loop" in RENDERERS
    svg = dqn_loop()
    lowered = svg.lower()
    assert svg.startswith("<svg")
    assert "optimal" not in lowered
    assert "policy" not in lowered
    for phrase in (
        "state",
        "online network",
        "Q-values",
        "ε-greedy choice",
        "chosen action",
        "current best guess",
        "environment",
        "reward + next state",
    ):
        assert phrase in svg


def test_talk_track_headings_match_builder():
    from talk_track import CHAPTERS

    nb = json.loads((ROOT / "colab_workshop.ipynb").read_text(encoding="utf-8"))
    joined = "\n".join("".join(c.get("source") or []) for c in nb["cells"])
    for row in CHAPTERS:
        assert row["heading"] in joined
        assert row["goal"] in joined
    ch5 = next(row for row in CHAPTERS if row["number"] == 5)
    assert ch5["slide_ids"] == (
        "section-dqn",
        "what-is-dqn",
        "dqn-q-values",
        "dqn-bellman",
        "dqn-replay",
        "dqn-target",
        "dqn-epsilon",
        "dqn-curve",
        "dqn-live-train",
        "dqn-live-eval",
        "dqn-bridge",
        "teacher-dataset",
    )
    for slide_id in (
        "dqn",
        "dqn-mario-intro",
        "dqn-analogy",
        "dqn-backup-analogy",
        "dqn-study-analogy",
        "dqn-math-overview",
        "dqn-explore-analogy",
        "dqn-mario-map",
        "dqn-mario",
    ):
        assert slide_id not in ch5["slide_ids"]
    ch6 = next(row for row in CHAPTERS if row["number"] == 6)
    assert ch6["slide_ids"] == (
        "section-packs",
        "synthetic-homework",
        "trace-to-pair",
        "dataset-filters",
        "why-warmstart",
    )


def test_talk_track_matches_presentation_and_new_dqn_slides():
    from talk_track import CHAPTERS

    deck = ROOT.parent / "SLM-RL-Presentation"
    slides = (deck / "src" / "data" / "slides.js").read_text(encoding="utf-8")
    track = (deck / "src" / "data" / "talkTrack.js").read_text(encoding="utf-8")
    ch5 = next(row for row in CHAPTERS if row["number"] == 5)
    for slide_id in ch5["slide_ids"]:
        assert f"id: '{slide_id}'" in slides
        assert f"'{slide_id}'" in track
    assert "Watch learning happen" in slides
    assert "Let the trained DQN play" in slides
    assert "play before you train" in slides.lower()
    assert "10,000" in slides


def _chapter_slice(joined: str, number: int, nxt: int) -> str:
    start = joined.index(f"## {number}.")
    end = joined.index(f"## {nxt}.")
    return joined[start:end]


def test_chapter_5_6_interactions_and_sequence(tmp_path):
    """Deck order, learner moments, live Mario, and trace inspection."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_nb", WORKSHOP / "build_nb.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    dest = tmp_path / "colab_workshop.ipynb"
    spec.loader.exec_module(mod)
    mod.OUT = dest
    mod.cells.clear()
    mod.main()
    joined = "\n".join("".join(c.get("source") or []) for c in json.loads(dest.read_text()).get("cells") or [])
    ch5 = _chapter_slice(joined, 5, 6)
    ch6 = _chapter_slice(joined, 6, 7)

    order = ("dqn-loop", "dqn-q-values", "dqn-bellman", "dqn-replay", "dqn-target")
    positions = [ch5.index(f"{name}.svg") for name in order]
    assert positions == sorted(positions)
    assert "dqn-loop.svg" in ch5

    assert "# @title Which move should Mario choose?" in ch5
    assert "higher number wins" in ch5
    assert "record_guess" in ch5
    assert "NEXT_Q_RIGHT" in ch5 and "GAMMA" in ch5 and "PREDICTED_Q" in ch5
    assert "EPSILON" in ch5 and "random.Random(SEED)" in ch5
    assert "# @title Predict the Mario clips" not in ch5
    assert "mario-untrained.mp4" not in ch5
    assert "# @title Play before you train" in joined
    assert joined.index("# @title Play before you train") < joined.index("## 2. Config")
    assert "INSTALL_MARIO" in joined
    eval_title = "# @title Watch the pretrained DQN play Mario"
    assert ch5.index("# @title Train Mario live") < ch5.index(eval_title)
    assert ch5.index(eval_title) < ch5.index("# @title Teacher knobs")
    assert ch5.index("dqn-encoders.svg") > ch5.index(eval_title)
    assert 'EVAL_SOURCE = "public-final"' in ch5
    assert "EVAL_STEPS = 10000" in ch5
    assert "TRAINING_MODE" in ch5
    assert "local-trained" in ch5

    assert "# @title Inspect a trace before SFT" in ch6
    assert ch6.index("select_episodes") < ch6.index("bake_pack")
    assert ch6.index("export_sft_dataset") < ch6.index("bake_pack")
    assert "not a training pair" in ch6
    assert "raw rollout row" in ch6
    assert "# @title Packs (optional hub)" in ch6


def test_fresh_runtime_form_cells_do_not_shadow_game_after_knobs():
    """GAME is set in knobs; later cells must not hard-reset it to boxing."""
    nb = json.loads((ROOT / "colab_workshop.ipynb").read_text(encoding="utf-8"))
    seen_knobs = False
    for cell in nb["cells"]:
        src = "".join(cell.get("source") or [])
        if src.startswith("# @title Workshop knobs"):
            seen_knobs = True
            continue
        if seen_knobs and cell["cell_type"] == "code":
            assert not any(line.startswith('GAME = "boxing"') for line in src.splitlines())
