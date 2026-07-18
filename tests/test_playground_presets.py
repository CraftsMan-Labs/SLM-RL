"""Plan 026 Phase D+E: HF side panel markers and /api/hardware presets.

Stdlib + pytest only. Hardware detection is real (detect_host on this
machine) but never loads a model; preset IDs are asserted against the
official-org allowlist in presets.py.
"""

from __future__ import annotations

import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from slm_rl.config.loader import load_tiers
from slm_rl.playground import experiments as exp_mod
from slm_rl.playground import server as pg_server_mod
from slm_rl.playground.page import PAGE
from slm_rl.playground.presets import (
    NEMOTRON_FLASH_1B,
    OFFICIAL_ORGS,
    TIER_PRESETS,
    hardware_payload,
    presets_for_tier,
)
from slm_rl.playground.tutorial_content import CARDS


class _ServerContext:
    def __init__(self, home: Path, game: str = "space-invaders"):
        handler_cls = pg_server_mod._make_handler(home, game)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_ServerContext":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5.0)

    def get(self, path: str) -> http.client.HTTPResponse:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", path)
        return conn.getresponse()


@pytest.fixture(autouse=True)
def _reset_locks():
    for kind in ("quick", "evolve", "theater", "bake"):
        exp_mod._ACTIVE[kind] = None
        exp_mod._ACTIVE_OWNER[kind] = None
    yield
    for kind in ("quick", "evolve", "theater", "bake"):
        exp_mod._ACTIVE[kind] = None
        exp_mod._ACTIVE_OWNER[kind] = None


# --- Phase D: signup side panel + HF copy ---------------------------------


def test_page_has_hf_side_panel_and_token_link():
    assert 'id="hf-panel"' in PAGE
    assert "https://huggingface.co/settings/tokens" in PAGE
    assert "hf-steps" in PAGE  # signup overlay steps
    assert "skip" in PAGE.lower()  # skip path still present


def test_signup_card_documents_create_account_and_token_url():
    body = CARDS["signup_card"]["body"]
    assert "huggingface.co" in body
    assert "https://huggingface.co/settings/tokens" in body
    assert "optional" in body.lower() or "Skip" in body


def test_model_field_card_rejects_qwen36_small():
    assert "Qwen3.6" in CARDS["model_field"]["body"]


# --- Phase E: presets + /api/hardware ------------------------------------


def test_every_preset_is_official_org_id():
    for tier_name, presets in TIER_PRESETS.items():
        for p in presets:
            org, _, rest = p.model.partition("/")
            assert org in OFFICIAL_ORGS, f"{tier_name}: {p.model!r} not official org"
            assert rest, f"{tier_name}: {p.model!r} missing repo name"
            assert "Qwen3.6" not in p.model


def test_nemotron_marked_experimental_in_label():
    from slm_rl.playground.presets import preset_label

    rows = presets_for_tier("any-8gb")
    nemo = next(r for r in rows if r["model"] == NEMOTRON_FLASH_1B)
    assert nemo["experimental"] is True
    assert "experimental" in nemo["label"].lower()
    assert "trust_remote_code" in nemo["label"]
    # label helper matches API row
    assert "experimental" in preset_label(
        next(p for p in TIER_PRESETS["any-8gb"] if p.model == NEMOTRON_FLASH_1B)
    ).lower()


def test_preset_defaults_match_hardware_yaml():
    """First preset per tier must equal configs/hardware.yaml default model."""
    tiers = {t.name: t for t in load_tiers()}
    for tier_name, presets in TIER_PRESETS.items():
        assert tier_name in tiers, f"preset tier {tier_name!r} missing from hardware.yaml"
        assert presets[0].model == tiers[tier_name].model, (
            f"{tier_name}: preset default {presets[0].model!r} != yaml {tiers[tier_name].model!r}"
        )


def test_mac_16gb_yaml_is_text_lfm_instruct():
    tier = next(t for t in load_tiers() if t.name == "mac-16gb")
    assert tier.model == "LiquidAI/LFM2.5-1.2B-Instruct"
    assert tier.backend == "transformers"
    assert tier.train == "grpo"


def test_hardware_payload_shape():
    payload = hardware_payload()
    assert payload["tier"] in TIER_PRESETS
    assert payload["model"]
    assert payload["backend"]
    assert isinstance(payload["presets"], list)
    assert payload["presets"]
    for row in payload["presets"]:
        org = row["model"].split("/", 1)[0]
        assert org in OFFICIAL_ORGS
    assert "host" in payload


def test_api_hardware_route(tmp_path: Path):
    with _ServerContext(tmp_path) as ctx:
        resp = ctx.get("/api/hardware")
        body = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 200
        assert body["tier"] in TIER_PRESETS
        ids = [p["model"] for p in body["presets"]]
        assert ids
        for mid in ids:
            assert mid.split("/", 1)[0] in OFFICIAL_ORGS
            assert "Qwen3.6" not in mid
        # Resolved tier's presets must match the table exactly (order + ids).
        expected = [p.model for p in TIER_PRESETS[body["tier"]]]
        assert ids == expected


def test_page_has_tier_banner_and_preset_select():
    assert 'id="tier-banner"' in PAGE
    assert 'id="f-preset"' in PAGE
    assert "/api/hardware" in PAGE
