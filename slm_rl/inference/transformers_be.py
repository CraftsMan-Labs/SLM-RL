"""transformers backend (CUDA / MPS / CPU). Requires the [cuda] extra.

Batched chat-template generation for small models, PEFT adapter hot-swap for
frozen/champion adapters, and mean per-token logprob for the entropy metric.
"""

from __future__ import annotations

import gc
from pathlib import Path

import torch

from slm_rl.inference.base import GenOutput, GenParams, InferenceBackend


def _pick_device(device: str | None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TransformersBackend(InferenceBackend):
    def __init__(self, model_id: str, four_bit: bool = False, device: str | None = None,
                 max_context: int = 2048):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = _pick_device(device)
        self.max_context = max_context
        self._adapters: set[str] = set()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: dict = {"dtype": torch.bfloat16 if self.device != "cpu" else torch.float32}
        if four_bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            kwargs["device_map"] = "cuda"
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(self.device)
        self.model.eval()

    def generate(self, chats, params: GenParams) -> list[GenOutput]:
        enc = self.tokenizer.apply_chat_template(
            chats, add_generation_prompt=True, tokenize=True, padding=True,
            truncation=True, max_length=self.max_context,
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
                metadata={"n_tokens": n},
            ))
        return results

    def load_adapter(self, adapter_path: Path | None) -> None:
        from peft import PeftModel

        if adapter_path is None:
            if isinstance(self.model, PeftModel):
                self.model.disable_adapter_layers()
            return

        name = str(adapter_path)
        if not isinstance(self.model, PeftModel):
            # first adapter: wrap the base model (PeftModel activates it)
            self.model = PeftModel.from_pretrained(self.model, adapter_path, adapter_name=name)
            self._adapters.add(name)
        else:
            if name not in self._adapters:
                self.model.load_adapter(adapter_path, adapter_name=name)
                self._adapters.add(name)
            self.model.enable_adapter_layers()
            self.model.set_adapter(name)

    def close(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _truncate_at_stop(text: str, stop: list[str]) -> str:
    for s in stop:
        idx = text.find(s)
        if idx != -1:
            text = text[:idx]
    return text
