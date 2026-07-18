"""HF token for Hub downloads (transformers + huggingface_hub).

Workshop tokens live in profile.json; CLI/Docker may set HF_TOKEN. Call
`apply_hf_token` once so ambient + explicit `token=hf_token()` both work.
Never log the return value.
"""

from __future__ import annotations

import os


def hf_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    return None


def apply_hf_token(token: str | None) -> str | None:
    """Put a non-empty token into the process env; return whatever is available."""
    value = (token or "").strip() or hf_token()
    if value:
        os.environ["HF_TOKEN"] = value
        os.environ["HUGGING_FACE_HUB_TOKEN"] = value
    return value
