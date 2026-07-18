"""Inference backend contract. Implementations (each in its own module,
heavy imports strictly lazy): TransformersBackend (CUDA/MPS/CPU),
MLXBackend, VLTransformersBackend ("transformers-vl" — vision demo).

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


@dataclass
class GenOutput:
    text: str
    logprob: float | None = None


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


def _truncate_at_stop(text: str, stop: list[str]) -> str:
    for s in stop:
        idx = text.find(s)
        if idx != -1:
            text = text[:idx]
    return text


def _pick_device(torch_mod: Any, device: str | None = None) -> str:
    if device:
        return device
    if torch_mod.cuda.is_available():
        return "cuda"
    if getattr(torch_mod.backends, "mps", None) and torch_mod.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_peft_adapter(model: Any, path: Path | None) -> Any:
    """Hot-swap a single LoRA adapter (None disables). Returns the model."""
    from peft import PeftModel

    if path is None:
        if isinstance(model, PeftModel):
            model.disable_adapter_layers()
        return model
    name = str(path)
    if not isinstance(model, PeftModel):
        return PeftModel.from_pretrained(model, path, adapter_name=name)
    if name not in model.peft_config:
        model.load_adapter(path, adapter_name=name)
    model.enable_adapter_layers()
    model.set_adapter(name)
    return model


def create_backend(name: str, model_id: str, quantization: str | None = None) -> InferenceBackend:
    """Factory resolving a tier's backend string to an implementation.
    Imports lazily so the core package works with no ML extras installed."""
    if name in ("transformers", "transformers-4bit"):
        from slm_rl.inference.transformers_be import TransformersBackend

        return TransformersBackend(model_id, four_bit=name.endswith("4bit"))
    if name == "mlx":
        from slm_rl.inference.mlx_be import MLXBackend

        return MLXBackend(model_id, quantization=quantization)
    if name == "transformers-vl":
        from slm_rl.inference.transformers_vl_be import VLTransformersBackend

        return VLTransformersBackend(model_id)
    raise ValueError(f"Unknown backend {name!r}")
