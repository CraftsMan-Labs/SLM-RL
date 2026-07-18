"""Local attendee profile: `<home>/playground/profile.json` `{name,
hf_username, hf_token, created_at}`. Plan 021 design decision 1.

NOT a multi-user account system -- each attendee runs their own playground
process on their own laptop, so "the" profile is a single file, not a
per-request session. Signup must work fully offline (name-only, token
skipped): `hf_username` is resolved lazily at first PUBLISH, never at
signup, so `save_profile` never makes a network call.

Token hygiene (CODING_GUIDELINE + plan 021 hard rule 3): written with file
mode 0600 via `os.open(..., mode=0o600)` (not chmod-after, which leaves a
window where the file is world-readable); `masked()` is the ONLY form
callers should ever put in an HTTP response, a log line, or an error
message -- the full token stays inside this module's return value from
`load_profile()` and whatever calls it directly for a publish.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


class InvalidProfile(Exception):
    """Bad signup payload (empty name, malformed token) -- mapped to HTTP 400."""


@dataclass
class Profile:
    name: str
    hf_username: str | None
    hf_token: str | None
    created_at: str

    def masked(self) -> dict:
        """Safe-to-serialize view: token reduced to its last 4 chars (or
        None if there isn't one). Never put `self.hf_token` itself in a
        dict that reaches an HTTP response, a log, or an error string."""
        token_display = None
        if self.hf_token:
            token_display = f"...{self.hf_token[-4:]}" if len(self.hf_token) > 4 else "...(short)"
        return {
            "name": self.name,
            "hf_username": self.hf_username,
            "has_token": bool(self.hf_token),
            "token_masked": token_display,
            "created_at": self.created_at,
        }


def profile_path(home: Path | str) -> Path:
    return Path(home) / "playground" / "profile.json"


def load_profile(home: Path | str) -> Profile | None:
    """None if no profile has been saved yet (the signup-gate 404 case)."""
    path = profile_path(home)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return Profile(
        name=data["name"],
        hf_username=data.get("hf_username"),
        hf_token=data.get("hf_token"),
        created_at=data["created_at"],
    )


def validate_signup(name: str, hf_token: str | None) -> None:
    if not name or not name.strip():
        raise InvalidProfile("name is required")
    if hf_token and not hf_token.startswith("hf_"):
        raise InvalidProfile("hf_token must look like an HF token (starts with 'hf_')")


def save_profile(
    home: Path | str, name: str, hf_token: str | None = None, hf_username: str | None = None
) -> Profile:
    """Validate + write. Token is optional -- signup must succeed with just
    a name (plan 021 design decision 3: no lockout). Never resolves
    `hf_username` here even if a token is given; that happens lazily at
    first publish (`resolve_username`) so signup stays offline."""
    validate_signup(name, hf_token)
    path = profile_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_profile(home)
    profile = Profile(
        name=name.strip(),
        hf_username=hf_username if hf_username is not None else (existing.hf_username if existing else None),
        hf_token=hf_token,
        created_at=existing.created_at if existing else datetime.now(timezone.utc).isoformat(),
    )
    _write(path, profile)
    return profile


def resolve_username(home: Path | str, profile: Profile) -> Profile:
    """Lazily fill `hf_username` from the HF Hub (`whoami`) and cache it
    back into profile.json. Called at first PUBLISH, not at signup (design
    decision 1) -- this is the one place in this module that touches the
    network. Callers pass `token=` explicitly (never ambient); any failure
    propagates to the caller unmasked-token-free (the exception carries no
    token text, only whatever huggingface_hub's own message says, which
    callers must still not log raw -- see hf_publish.py)."""
    if profile.hf_username or not profile.hf_token:
        return profile
    from huggingface_hub import HfApi

    who = HfApi().whoami(token=profile.hf_token)
    username = who.get("name") if isinstance(who, dict) else None
    if not username:
        return profile
    profile.hf_username = username
    _write(profile_path(home), profile)
    return profile


def _write(path: Path, profile: Profile) -> None:
    """0600 from creation, never chmod-after (a chmod-after window is a
    real race on a shared/multi-user filesystem, however unlikely on a
    workshop laptop)."""
    payload = json.dumps(asdict(profile), indent=2).encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(payload)
