"""Plan 023: tutorial mode (info cards + intro panel + toggle).

Stdlib + pytest only -- no model loads, no subprocess, no network. Covers:
  - every slm_rl.playground.knobs.KNOBS entry has a tutorial_content.CARDS
    entry (looped programmatically, per plan: a future knob without a card
    must fail loudly, not silently pass a hardcoded list)
  - required non-knob card coverage (teacher select, model/backend fields,
    reward tab, evolve/watch/A-B/gens/publish, scoreboard columns, signup)
  - marker tests: intro panel present, tutorial toggle present, the
    gate-purity doctrine string present VERBATIM, reward-tab card mentions
    shape_reward(ctx)
  - every card body is <= 80 words (plan design decision 2)
"""

from __future__ import annotations

from slm_rl.playground.knobs import KNOBS
from slm_rl.playground.page import PAGE
from slm_rl.playground.tutorial_content import CARDS, GATE_PURITY_DOCTRINE

# --- Coverage: every KNOBS entry has a card (loop, not a hardcoded list) ---


def test_every_knob_has_a_tutorial_card():
    missing = [knob.key for knob in KNOBS if knob.key not in CARDS]
    assert not missing, f"knobs missing tutorial cards: {missing}"


def test_every_knob_card_appears_in_the_page_via_knob_key_in_tutorial_json():
    # Knob fields are built client-side (knobField() in page.py's JS) from
    # the /api/knobs schema, so their cards live in the embedded tutorial
    # JSON blob (read by JS at init), not as static data-card="..." markup
    # the way static fields are. Assert each knob key round-trips through
    # the blob actually embedded in PAGE.
    for knob in KNOBS:
        needle = f'"{knob.key}"'
        assert needle in PAGE, f"knob {knob.key!r} card missing from embedded tutorial JSON"


# --- Coverage: required non-knob regions (design decision 3) --------------


_REQUIRED_NON_KNOB_CARDS = [
    "teacher_select",
    "model_field",
    "backend_field",
    "reward_tab",
    "evolve_button",
    "watch_link",
    "ab_button",
    "gens_link",
    "play_again_button",
    "publish_button",
    "signup_card",
    "scoreboard_name",
    "scoreboard_model",
    "scoreboard_episodes",
    "scoreboard_mean",
    "scoreboard_median",
    "scoreboard_max",
    "scoreboard_actions",
    "scoreboard_interventions",
    "scoreboard_status",
]


def test_required_non_knob_cards_present_in_content_module():
    missing = [key for key in _REQUIRED_NON_KNOB_CARDS if key not in CARDS]
    assert not missing, f"required cards missing from tutorial_content.CARDS: {missing}"


def test_required_static_cards_wired_into_page_markup():
    # These regions are static HTML (not client-built like the knob grid /
    # scoreboard rows), so their data-card="<key>" attribute is literally in
    # PAGE -- a stronger check than "the card text exists somewhere".
    static_keys = [
        "teacher_select",
        "model_field",
        "backend_field",
        "reward_tab",
        "evolve_button",
        "watch_link",
        "ab_button",
        "gens_link",
        "play_again_button",
        "publish_button",
        "signup_card",
        "scoreboard_name",
        "scoreboard_model",
        "scoreboard_episodes",
        "scoreboard_mean",
        "scoreboard_median",
        "scoreboard_max",
        "scoreboard_actions",
        "scoreboard_interventions",
        "scoreboard_status",
    ]
    for key in static_keys:
        assert f'data-card="{key}"' in PAGE, f"no (i) icon wired for {key!r}"


# --- Card body word-count ceiling (design decision 2: <= 80 words each) ---


def test_every_card_body_is_at_most_80_words():
    over = {key: len(card["body"].split()) for key, card in CARDS.items() if len(card["body"].split()) > 80}
    assert not over, f"cards over the 80-word limit: {over}"


def test_every_card_has_title_and_body():
    for key, card in CARDS.items():
        assert card.get("title"), f"card {key!r} missing a title"
        assert card.get("body"), f"card {key!r} missing a body"


# --- Marker tests: intro panel, toggle, gate-purity doctrine, ctx keys ----


def test_intro_panel_present():
    assert 'id="intro-panel"' in PAGE
    assert "How it works" in PAGE
    # the ROLLOUT -> DATASET -> TRAIN -> EVAL -> GATE diagram from README.md
    assert "ROLLOUT" in PAGE
    assert "DATASET" in PAGE
    assert "TRAIN" in PAGE
    assert "EVAL" in PAGE
    assert "GATE" in PAGE


def test_tutorial_toggle_present():
    assert 'id="tutorial-checkbox"' in PAGE
    assert 'id="tutorial-toggle"' in PAGE


def test_gate_purity_doctrine_present_verbatim():
    # Canonical phrasing (docs/HYBRID_RL.md, CODING_GUIDELINE.md invariant
    # 2) -- must appear character-for-character, not paraphrased.
    assert GATE_PURITY_DOCTRINE == "Steering must never be counted as model improvement."
    assert GATE_PURITY_DOCTRINE in PAGE
    assert GATE_PURITY_DOCTRINE in CARDS["reward_tab"]["body"]


def test_reward_tab_card_mentions_shape_reward_ctx():
    assert "shape_reward(ctx)" in CARDS["reward_tab"]["body"]


# --- Spot-check markup for a knob, the reward tab, publish, A/B (accept.2) -


def test_page_spot_check_curl_style_markers():
    # Mirrors the plan's acceptance curl+grep check: card markup present for
    # a knob (via the embedded JSON, since knob fields are client-built),
    # the reward tab, publish, and A/B -- run in-process instead of over
    # HTTP for speed, same assertions a curl+grep session would make.
    assert '"max_turns"' in PAGE  # a knob, embedded in tutorial-data JSON
    assert 'data-card="reward_tab"' in PAGE
    assert 'data-card="publish_button"' in PAGE
    assert 'data-card="ab_button"' in PAGE
