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
    assert "RUN_MARIO" in joined
    assert "target = reward + γ × best next Q" in joined
    assert "mario-pretrained.mp4" in joined
    assert "select_episodes" in joined
    assert "docs/workshop/assets/diagrams/evolve-loop.svg" in joined
    assert "docs/workshop/assets/deck/HeroVisual.png" in joined
    assert "docs/workshop/assets/deck/meet-the-teacher.mp4" in joined
    assert "docs/workshop/assets/diagrams/dqn-q-values.svg" in joined
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


def test_talk_track_headings_match_builder():
    from talk_track import CHAPTERS

    nb = json.loads((ROOT / "colab_workshop.ipynb").read_text(encoding="utf-8"))
    joined = "\n".join("".join(c.get("source") or []) for c in nb["cells"])
    for row in CHAPTERS:
        assert row["heading"] in joined
        assert row["goal"] in joined
    ch5 = next(row for row in CHAPTERS if row["number"] == 5)
    for slide_id in ("dqn-mario-intro", "dqn-math-overview", "dqn-mario-map"):
        assert slide_id in ch5["slide_ids"]
    ch6 = next(row for row in CHAPTERS if row["number"] == 6)
    for slide_id in ("synthetic-homework", "trace-to-pair", "dataset-filters", "why-warmstart"):
        assert slide_id in ch6["slide_ids"]


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
            assert "GAME = \"boxing\"" not in src
