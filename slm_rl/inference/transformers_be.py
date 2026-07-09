"""transformers backend (CUDA / MPS / CPU). Requires the [cuda] extra."""

from __future__ import annotations

from slm_rl.inference.base import GenOutput, GenParams, InferenceBackend


class TransformersBackend(InferenceBackend):
    def __init__(self, model_id: str, four_bit: bool = False):
        raise NotImplementedError("Phase 1")

    def generate(self, chats, params: GenParams) -> list[GenOutput]:
        raise NotImplementedError("Phase 1")
