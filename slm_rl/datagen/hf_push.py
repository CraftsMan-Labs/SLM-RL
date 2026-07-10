"""Push a generation's dataset artifacts to a Hugging Face dataset repo.

Layout in the repo mirrors the run tree so multiple runs/generations coexist:
    <run_id>/gen_000/eval/results.json
    <run_id>/gen_001/rollouts/*.jsonl
    <run_id>/gen_001/dataset/{train.parquet, grpo.jsonl|sft.jsonl}
    <run_id>/gen_001/{metrics.json, MANIFEST.json}

Auth: uses the cached `hf auth login` token or HF_TOKEN. Uploads are
best-effort — callers in the training loop must not die on network failure.
"""

from __future__ import annotations

from pathlib import Path

# datasets only, deliberately: adapters/trainer state are model artifacts
_PATTERNS = ["rollouts/*.jsonl", "dataset/*", "eval/*.json", "*.json"]


def push_generation(
    repo_id: str, run_id: str, generation: int, gen_dir: Path, private: bool = True
) -> str:
    """Upload one generation's data artifacts; returns the commit URL."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    info = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(gen_dir),
        path_in_repo=f"{run_id}/gen_{generation:03d}",
        allow_patterns=_PATTERNS,
        commit_message=f"{run_id}: generation {generation}",
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
