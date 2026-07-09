"""MLX backend for Apple Silicon. Requires the [mac] extra."""

from __future__ import annotations

from slm_rl.inference.base import GenOutput, GenParams, InferenceBackend


class MLXBackend(InferenceBackend):
    def __init__(self, model_id: str, quantization: str | None = "q4"):
        raise NotImplementedError("Phase 1")

    def generate(self, chats, params: GenParams) -> list[GenOutput]:
        raise NotImplementedError("Phase 1")
