"""Vision-language transformers backend (CUDA / MPS / CPU). Requires the
[cuda] extra (torch + transformers, same as transformers_be.py, PLUS
pillow + torchvision -- caught by the real-model smoke, plan 011:
`Lfm2VlImageProcessor` raises ImportError without torchvision even though
transformers itself imports fine without it; pyproject.toml's `cuda` extra
now pins both).

Mirrors TransformersBackend's contract (GenParams in, GenOutput out) but
`generate(chats, params)` accepts chats whose message `content` is a LIST
of parts (`{"type": "text", "text": ...}` / `{"type": "image", "image":
<PIL.Image>}`) instead of a plain string, and loads via
`AutoModelForImageTextToText` + `AutoProcessor` instead of
`AutoModelForCausalLM` + `AutoTokenizer`.

Verified live against the HF Hub 2026-07 (plan 011): `LiquidAI/LFM2.5-VL-
450M` exists as a transformers-native repo (library_name=transformers,
architectures=["Lfm2VlForConditionalGeneration"], model_type="lfm2_vl",
siblings include config.json/model.safetensors/processor_config.json --
not GGUF/ONNX/MLX-only). `LiquidAI/LFM2.5-VL-1.6B` also exists the same
way. The model card's documented usage (fetched 2026-07) is exactly the
AutoModelForImageTextToText + AutoProcessor + `processor.apply_chat_template
(conversation, add_generation_prompt=True, return_tensors="pt",
return_dict=True, tokenize=True)` pattern this backend follows -- 450M is
the default (smaller footprint, matches the roadmap's expectation), 1.6B
is a drop-in `--model` override (both fit well inside the 16GB presenter
box in bf16: 450M ~1GB, 1.6B ~4GB).

`torch` is imported lazily inside methods, not at module top (unlike
transformers_be.py) -- deliberately, so this module can be imported and
unit-tested with `torch`/`transformers` fully mocked via sys.modules
injection (same discipline as mlx_be.py) without the
[cuda] extra installed. Hard rule: pytest never loads a real model.
"""

from __future__ import annotations

from pathlib import Path

from slm_rl.inference.base import (
    GenOutput,
    GenParams,
    InferenceBackend,
    _pick_device,
    _truncate_at_stop,
    load_peft_adapter,
)

DEFAULT_VL_MODEL = "LiquidAI/LFM2.5-VL-450M"


class VLTransformersBackend(InferenceBackend):
    """`create_backend` name `"transformers-vl"`. One model resident at a
    time, same 8GB-budget discipline as every other backend."""

    def __init__(self, model_id: str = DEFAULT_VL_MODEL):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from slm_rl.hf_auth import hf_token

        self.device = _pick_device(torch)
        token = hf_token()

        self.processor = AutoProcessor.from_pretrained(model_id, token=token)

        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, token=token,
        ).to(self.device)
        self.model.eval()

    def generate(self, chats: list[list[dict]], params: GenParams) -> list[GenOutput]:
        import torch

        # No batching: the processor's chat template + image preprocessing
        # doesn't left-pad multi-image batches cleanly across model
        # families, and the demo's rollout_batch_size stays 1 (plan 011
        # scope is rollout+viewer, not throughput) -- same serial contract
        # llama.cpp/MLX already use (ponytail: batched VL generation would
        # need per-sample image-count-aware padding).
        results = []
        for messages in chats:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
            ).to(self.model.device)

            do_sample = params.temperature > 0
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    do_sample=do_sample,
                    temperature=params.temperature if do_sample else None,
                    top_p=params.top_p if do_sample else None,
                    max_new_tokens=params.max_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            prompt_len = inputs["input_ids"].shape[1]
            gen_ids = out.sequences[:, prompt_len:]
            scores = self.model.compute_transition_scores(
                out.sequences, out.scores, normalize_logits=True
            )
            text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
            text = _truncate_at_stop(text, params.stop)

            n = gen_ids.shape[1]
            logprob = float(scores[0][:n].mean()) if n else None
            results.append(GenOutput(text=text, logprob=logprob))
        return results

    def load_adapter(self, adapter_path: Path | None) -> None:
        self.model = load_peft_adapter(self.model, adapter_path)

    def close(self) -> None:
        import gc

        import torch

        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
