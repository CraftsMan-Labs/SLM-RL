"""Shared LoRA helpers: target modules, bootstrap, teardown, log metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# LFM2/LFM2.5 is a hybrid conv+attention arch: attention q/k/v + out_proj,
# conv in_proj, GLU w1/w2/w3 (per Unsloth's LFM2.5 guide — note out_proj, not
# o_proj). Substring "lfm2" matches "lfm2.5" too.
LORA_TARGETS = {
    "lfm2": ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def target_modules_for(model_id: str) -> list[str] | str:
    lower = model_id.lower()
    for key, targets in LORA_TARGETS.items():
        if key in lower:
            return targets
    return "all-linear"  # PEFT fallback for unknown architectures


def bf16_ok() -> bool:
    """Ampere+ only. T4/V100 (Colab free tier) must use fp16."""
    import torch
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def compute_dtype(cuda: bool):
    import torch
    if not cuda:
        return torch.float32  # CPU/MPS unchanged - MPS fp16 overflows on this stack
    return torch.bfloat16 if bf16_ok() else torch.float16


def _mps_available(torch: Any) -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def load_causal_lm(model_id: str, *, cuda: bool, four_bit: bool = False) -> Any:
    """Load a causal LM without wedging Apple MPS.

    HuggingFace/TRL parallel weight materialization onto MPS contends inside
    MetalShaderLibrary and can sit at ``Loading weights: 0/N`` for 40+ minutes.
    Always materialize on CPU first, then move once (single-threaded on MPS).
    """
    import torch
    from transformers import AutoModelForCausalLM

    from slm_rl.hf_auth import hf_token

    if four_bit and not cuda:
        import logging
        logging.getLogger(__name__).warning(
            "4-bit quantization needs CUDA (bitsandbytes has no CPU build); using full precision"
        )
        four_bit = False

    if four_bit:
        from transformers import BitsAndBytesConfig

        return AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype(True),
                bnb_4bit_use_double_quant=True,
            ),
            device_map="cuda",
            token=hf_token(),
        )

    dtype = compute_dtype(cuda)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
        token=hf_token(),
    )
    if cuda:
        return model.to("cuda")
    if _mps_available(torch):
        prev = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            return model.to("mps")
        finally:
            torch.set_num_threads(prev)
    return model


def bootstrap_lora(
    model_id: str, cfg: Any, init_adapter: Path | None, cuda: bool,
    four_bit: bool = False,
) -> tuple[Any, Any]:
    """Load a trainable PeftModel from `init_adapter`, or return
    `(base_model, LoraConfig)` for a fresh LoRA. Shared by reject_sft + grpo.

    Always returns a concrete model object (never a bare model_id string) so
    TRL cannot re-enter the hanging MPS ``from_pretrained`` path.
    """
    from peft import LoraConfig, PeftModel

    base = load_causal_lm(model_id, cuda=cuda, four_bit=four_bit)
    if four_bit and cuda:
        from peft import prepare_model_for_kbit_training
        base = prepare_model_for_kbit_training(base)
    if init_adapter is not None:
        return PeftModel.from_pretrained(base, str(init_adapter), is_trainable=True), None
    peft_config = LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=target_modules_for(model_id),
    )
    return base, peft_config


def release_trainer_memory(cuda: bool) -> None:
    """gc + optional CUDA cache clear after `del trainer, model` in the caller."""
    import gc

    gc.collect()
    if cuda:
        import torch

        torch.cuda.empty_cache()


def last_log_metrics(logs: list[dict], keys: tuple[str, ...]) -> dict[str, Any]:
    """Last logged value for each key present in TRL `log_history`."""
    metrics: dict[str, Any] = {}
    for key in keys:
        vals = [row[key] for row in logs if key in row]
        if vals:
            metrics[key] = vals[-1]
    return metrics
