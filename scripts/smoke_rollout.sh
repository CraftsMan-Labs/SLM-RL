#!/usr/bin/env bash
# CPU smoke test (Phase 1): random-agent rollouts -> JSONL -> parquet.
set -euo pipefail
uv run slm-rl rollout --game mastermind --agent random --episodes 5
