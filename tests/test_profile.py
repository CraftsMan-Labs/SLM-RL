"""Plan 021: the local attendee profile store. Stdlib-only, no network --
`resolve_username` is the one function that touches the hub and it's
mocked here (never real huggingface_hub calls in pytest)."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from slm_rl.playground.profile import (
    InvalidProfile,
    load_profile,
    profile_path,
    resolve_username,
    save_profile,
)


def test_save_then_load_roundtrips(tmp_path: Path):
    save_profile(tmp_path, name="Ada", hf_token="hf_abcdefghij")
    profile = load_profile(tmp_path)
    assert profile is not None
    assert profile.name == "Ada"
    assert profile.hf_token == "hf_abcdefghij"
    assert profile.hf_username is None
    assert profile.created_at


def test_load_returns_none_before_any_signup(tmp_path: Path):
    assert load_profile(tmp_path) is None


def test_file_mode_is_exactly_0600(tmp_path: Path):
    save_profile(tmp_path, name="Ada", hf_token="hf_abcdefghij")
    path = profile_path(tmp_path)
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600, f"expected 0600, got {oct(mode)}"


def test_profile_path_is_under_home_playground_never_repo(tmp_path: Path):
    save_profile(tmp_path, name="Ada")
    path = profile_path(tmp_path)
    assert path == tmp_path / "playground" / "profile.json"
    # never inside the repo working tree / anything git could pick up
    repo_root = Path(__file__).resolve().parent.parent
    assert repo_root not in path.parents


def test_name_only_signup_is_valid_no_lockout(tmp_path: Path):
    profile = save_profile(tmp_path, name="Grace", hf_token=None)
    assert profile.name == "Grace"
    assert profile.hf_token is None
    loaded = load_profile(tmp_path)
    assert loaded is not None
    assert loaded.hf_token is None


def test_empty_name_rejected(tmp_path: Path):
    with pytest.raises(InvalidProfile):
        save_profile(tmp_path, name="   ")


def test_bad_token_prefix_rejected(tmp_path: Path):
    with pytest.raises(InvalidProfile):
        save_profile(tmp_path, name="Ada", hf_token="not-a-real-token")


def test_masked_never_exposes_full_token(tmp_path: Path):
    profile = save_profile(tmp_path, name="Ada", hf_token="hf_abcdefghijklmnop")
    masked = profile.masked()
    assert "hf_abcdefghijklmnop" not in json.dumps(masked)
    assert masked["token_masked"] == "...mnop"
    assert masked["has_token"] is True
    assert "hf_token" not in masked


def test_masked_no_token_case(tmp_path: Path):
    profile = save_profile(tmp_path, name="Ada")
    masked = profile.masked()
    assert masked["has_token"] is False
    assert masked["token_masked"] is None


def test_resave_preserves_created_at_and_cached_username(tmp_path: Path, monkeypatch):
    save_profile(tmp_path, name="Ada", hf_token="hf_abcdefghij")
    profile = load_profile(tmp_path)

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def whoami(self, token=None):
            return {"name": "ada-hf"}

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    resolved = resolve_username(tmp_path, profile)
    assert resolved.hf_username == "ada-hf"

    first_created_at = load_profile(tmp_path).created_at
    # re-save (e.g. editing the token) must not reset created_at, and must
    # keep the cached username since the new save doesn't pass one.
    save_profile(tmp_path, name="Ada", hf_token="hf_newtoken1234")
    reloaded = load_profile(tmp_path)
    assert reloaded.created_at == first_created_at
    assert reloaded.hf_username == "ada-hf"


def test_resolve_username_is_noop_without_token(tmp_path: Path):
    profile = save_profile(tmp_path, name="Grace", hf_token=None)
    resolved = resolve_username(tmp_path, profile)
    assert resolved.hf_username is None


def test_resolve_username_is_noop_if_already_cached(tmp_path: Path, monkeypatch):
    profile = save_profile(tmp_path, name="Ada", hf_token="hf_abcdefghij", hf_username="cached")

    calls = []

    class FakeApi:
        def __init__(self, token=None):
            calls.append(token)

        def whoami(self, token=None):  # pragma: no cover - should never be called
            raise AssertionError("whoami should not be called when username already cached")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    resolved = resolve_username(tmp_path, profile)
    assert resolved.hf_username == "cached"
    assert calls == []
