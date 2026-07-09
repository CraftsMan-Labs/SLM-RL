"""vLLM backend (CUDA rollouts, multi-LoRA opponent serving).
Requires the [vllm] extra. Never used on 8GB tiers."""

from __future__ import annotations

from slm_rl.inference.base import GenOutput, GenParams, InferenceBackend


class VLLMBackend(InferenceBackend):
    def __init__(self, model_id: str):
        raise NotImplementedError("Phase 2")

    def generate(self, chats, params: GenParams) -> list[GenOutput]:
        raise NotImplementedError("Phase 2")
