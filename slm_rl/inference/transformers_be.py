"""transformers backend (CUDA / MPS / CPU). Requires the [cuda] extra.

Batched chat-template generation for small models, PEFT adapter hot-swap for
frozen/champion adapters, and mean per-token logprob for the entropy metric.
"""

from __future__ import annotations

import gc
from pathlib import Path

import torch

from slm_rl.inference.base import (
    GenOutput,
    GenParams,
    InferenceBackend,
    _pick_device,
    _truncate_at_stop,
    load_peft_adapter,
)

_MAX_CONTEXT = 2048


class TransformersBackend(InferenceBackend):
    def __init__(self, model_id: str, four_bit: bool = False):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from slm_rl.hf_auth import hf_token

        self.device = _pick_device(torch)
        token = hf_token()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # CPU-first then .to(device): parallel HF→MPS materialization wedges
        # MetalShaderLibrary ("Loading weights: 0%" for tens of minutes).
        kwargs: dict = {
            "dtype": torch.bfloat16 if self.device != "cpu" else torch.float32,
            "token": token,
        }
        if four_bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            kwargs["device_map"] = "cuda"
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        else:
            kwargs["low_cpu_mem_usage"] = True
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            if self.device != "cpu":
                if self.device == "mps":
                    prev = torch.get_num_threads()
                    torch.set_num_threads(1)
                    try:
                        self.model = self.model.to(self.device)
                    finally:
                        torch.set_num_threads(prev)
                else:
                    self.model = self.model.to(self.device)
        self.model.eval()

    def generate(self, chats, params: GenParams) -> list[GenOutput]:
        enc = self.tokenizer.apply_chat_template(
            chats, add_generation_prompt=True, tokenize=True, padding=True,
            truncation=True, max_length=_MAX_CONTEXT,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)

        do_sample = params.temperature > 0
        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                do_sample=do_sample,
                temperature=params.temperature if do_sample else None,
                top_p=params.top_p if do_sample else None,
                max_new_tokens=params.max_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True, output_scores=True,
            )

        gen_ids = out.sequences[:, enc["input_ids"].shape[1]:]
        scores = self.model.compute_transition_scores(
            out.sequences, out.scores, normalize_logits=True
        )  # (batch, gen_len), log-probs of chosen tokens
        texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        results = []
        pad_id = self.tokenizer.pad_token_id
        for i, text in enumerate(texts):
            mask = gen_ids[i] != pad_id
            n = int(mask.sum())
            logprob = float(scores[i][mask].mean()) if n else None
            results.append(GenOutput(
                text=_truncate_at_stop(text, params.stop),
                logprob=logprob,
            ))
        return results

    def load_adapter(self, adapter_path: Path | None) -> None:
        self.model = load_peft_adapter(self.model, adapter_path)

    def close(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
