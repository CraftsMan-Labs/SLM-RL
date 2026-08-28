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
    clamp_float,
    clamp_int,
    ensure_card,
    grade,
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
    assert 'GAME = "boxing"' in joined
    assert "PREDICT_PARSE" in joined
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
    assert "target = reward + γ × best next Q" in joined
    assert "mario-pretrained.mp4" in joined
    assert "select_episodes" in joined
    assert "docs/workshop/assets/diagrams/evolve-loop.svg" in joined
    assert "docs/workshop/assets/deck/HeroVisual.png" in joined
    assert "docs/workshop/assets/deck/meet-the-teacher.mp4" in joined
    assert "docs/workshop/assets/diagrams/dqn-q-values.svg" in joined
    assert "docs/workshop/assets/diagrams/dqn-loop.svg" in joined
    assert "PREDICT_ACTION" in joined
    assert "NEXT_Q_RIGHT" in joined
    assert "EPSILON" in joined
    assert "PREDICT_CLIP" in joined
    assert "export_sft_dataset" in joined
    assert "SKIP_GATES" not in joined
    assert "skip=SKIP_GATES" not in joined
    assert "# @title Join the room" in joined
    assert "ask(" in joined
    assert "show_card(CARD)" in joined
    assert "Runtime → Run all" in joined


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
        assert name in joined
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
    """Deck order, learner moments, dqn-loop, clips gate, and trace inspection."""
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
    assert ch5.index("dqn-encoders.svg") > ch5.index("mario-pretrained.mp4")

    assert "# @title Predict RIGHT vs RIGHT+A" in ch5
    assert "record_guess" in ch5
    assert "NEXT_Q_RIGHT" in ch5 and "GAMMA" in ch5 and "PREDICTED_Q" in ch5
    assert "EPSILON" in ch5 and "random.Random(SEED)" in ch5
    assert ch5.index("Which clip travels farthest") < ch5.index("mario-untrained.mp4")
    assert ch5.index("ask(") < ch5.index("mario-untrained.mp4")
    assert "# @title Play before you train" in joined
    assert joined.index("# @title Play before you train") < joined.index("## 2. Config")
    assert "INSTALL_MARIO" in joined
    assert ch5.index("mario-pretrained.mp4") < ch5.index("# @title Train Mario live")
    assert ch5.index("# @title Train Mario live") < ch5.index("# @title Evaluate the trained DQN")
    assert ch5.index("# @title Evaluate the trained DQN") < ch5.index("# @title Teacher knobs")
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
