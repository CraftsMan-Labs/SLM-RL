"""Inference backend contract. Implementations (each in its own module,
heavy imports strictly lazy): TransformersBackend (CUDA/MPS/CPU),
VLLMBackend, LlamaCppBackend (GGUF — the 8GB default), MLXBackend.

One model resident at a time: frozen-generation opponents are served by
`load_adapter` hot-swaps, never a second model copy (8GB budget rule).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GenParams:
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    stop: list[str] = field(default_factory=list)
    grammar: str | None = None  # constrained decoding, if supported


@dataclass
class GenOutput:
    text: str
    logprob: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class InferenceBackend(ABC):
    @abstractmethod
    def generate(self, chats: list[list[dict[str, Any]]], params: GenParams) -> list[GenOutput]: ...

    def load_adapter(self, adapter_path: Path | None) -> None:
        """Hot-swap a LoRA adapter (None = base model). Backends without
        adapter support raise."""
        raise NotImplementedError(f"{type(self).__name__} does not support adapters")

    def close(self) -> None:
        """Free the model / GPU memory. The single-GPU rollout->train handoff
        relies on this. No-op by default."""

    @property
    def supports_constrained_decoding(self) -> bool:
        return False


def create_backend(name: str, model_id: str, quantization: str | None = None) -> InferenceBackend:
    """Factory resolving a tier's backend string to an implementation.
    Imports lazily so the core package works with no ML extras installed."""
    if name in ("transformers", "transformers-4bit"):
        from slm_rl.inference.transformers_be import TransformersBackend

        return TransformersBackend(model_id, four_bit=name.endswith("4bit"))
    if name == "vllm":
        from slm_rl.inference.vllm_be import VLLMBackend

        return VLLMBackend(model_id)
    if name == "llama_cpp":
        from slm_rl.inference.llama_cpp_be import LlamaCppBackend

        return LlamaCppBackend(model_id, quantization=quantization)
    if name == "mlx":
        from slm_rl.inference.mlx_be import MLXBackend

        return MLXBackend(model_id, quantization=quantization)
    raise ValueError(f"Unknown backend {name!r}")
