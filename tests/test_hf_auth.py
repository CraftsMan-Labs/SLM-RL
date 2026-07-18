"""HF token helpers: env read + apply. Never hits the network."""

from __future__ import annotations

from slm_rl.hf_auth import apply_hf_token, hf_token


def test_hf_token_reads_env_and_skips_empty(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert hf_token() is None

    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "  ")
    assert hf_token() is None

    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_from_hub_var")
    assert hf_token() == "hf_from_hub_var"

    monkeypatch.setenv("HF_TOKEN", "hf_primary")
    assert hf_token() == "hf_primary"


def test_apply_hf_token_overwrites_empty_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "")
    assert apply_hf_token("hf_profiletoken12") == "hf_profiletoken12"
    assert hf_token() == "hf_profiletoken12"
