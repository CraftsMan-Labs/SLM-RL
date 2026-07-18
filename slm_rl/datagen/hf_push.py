"""Push a generation's dataset artifacts to a Hugging Face dataset repo.

Layout in the repo mirrors the run tree so multiple runs/generations coexist:
    <run_id>/gen_000/eval/results.json
    <run_id>/gen_001/rollouts/*.jsonl
    <run_id>/gen_001/dataset/{train.parquet, grpo.jsonl|sft.jsonl}
    <run_id>/gen_001/{metrics.json, MANIFEST.json}

Auth: `token=None` (the default) uses the cached `hf auth login` token or
HF_TOKEN, same as always. Callers that hold an explicit attendee token
(plan 021's publish flow) pass `token=...` so the upload goes to THAT
account rather than whatever's ambient on the machine — never mix the two.
Uploads are best-effort — callers in the training loop must not die on
network failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# datasets only, deliberately: adapters/trainer state are model artifacts
_PATTERNS = ["rollouts/*.jsonl", "dataset/*", "eval/*.json", "*.json"]


def create_and_upload_folder(
    api: Any,
    repo_id: str,
    *,
    repo_type: str,
    folder_path: Path | str,
    path_in_repo: str,
    commit_message: str,
    private: bool = True,
    allow_patterns: list[str] | None = None,
) -> Any:
    """create_repo + upload_folder. `api` must already carry `token=` on
    construction — do not pass token= again on these calls."""
    api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "folder_path": str(folder_path),
        "path_in_repo": path_in_repo,
        "commit_message": commit_message,
    }
    if allow_patterns is not None:
        kwargs["allow_patterns"] = allow_patterns
    return api.upload_folder(**kwargs)


def push_generation(
    repo_id: str,
    run_id: str,
    generation: int,
    gen_dir: Path,
    private: bool = True,
    token: str | None = None,
) -> str:
    """Upload one generation's data artifacts; returns the commit URL."""
    from huggingface_hub import HfApi

    info = create_and_upload_folder(
        HfApi(token=token),
        repo_id,
        repo_type="dataset",
        folder_path=gen_dir,
        path_in_repo=f"{run_id}/gen_{generation:03d}",
        commit_message=f"{run_id}: generation {generation}",
        private=private,
        allow_patterns=_PATTERNS,
    )
    return info.commit_url


def try_push_generation(repo_id: str, run_id: str, generation: int, gen_dir: Path) -> str | None:
    """push_generation that warns instead of raising — the evolve loop must
    survive HF being down or the token missing."""
    try:
        return push_generation(repo_id, run_id, generation, gen_dir)
    except Exception as e:  # noqa: BLE001 - any hub/network error is non-fatal here
        import warnings

        warnings.warn(
            f"HF dataset push failed for gen {generation}: {e} "
            "(set HF_TOKEN or run `hf auth login`; data remains on disk)",
            stacklevel=2,
        )
        return None
