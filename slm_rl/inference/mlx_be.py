"""MLX backend for Apple Silicon. Requires the [mac] extra.

IMPLEMENTED 2026-07, NOT YET VERIFIED ON REAL APPLE SILICON. mlx_lm is
arm64/darwin-only and CI (this repo's linux worktree/CI) cannot import or
run it — unit tests below mock the `mlx_lm` module entirely (import
injection, same pattern as other backend unit tests). Verify this module
end-to-end in the pre-workshop Mac pass: `uv sync --extra mac --extra dev`
then a real `slm-rl rollout --backend mlx` smoke on an actual Mac. Until
that pass, treat this as best-effort-correct-against-the-documented-API,
not proven.

mlx_lm.load() accepts model_id directly (community *-MLX-4bit snapshots or
on-the-fly convert via mlx_lm.convert).

Known gap (out of scope here, tracked for Phase 4): trainers write PEFT
adapters; mlx_lm's native LoRA loader expects its own format. No converter
exists yet, so `load_adapter` is NOT implemented — use transformers for
multi-generation evolve.
"""

from __future__ import annotations

from slm_rl.inference.base import GenOutput, GenParams, InferenceBackend, _truncate_at_stop


class MLXBackend(InferenceBackend):
    def __init__(self, model_id: str, quantization: str | None = "q4"):
        from mlx_lm import load

        # ponytail: quantization unused — mlx_lm resolves/quantizes from the
        # repo; keep the arg so create_backend(tier) call sites stay uniform.
        # mlx_lm.load has no token=; Hub auth comes from HF_TOKEN in the env
        # (set by playground profile / _hf_child_env / docker-compose).
        _ = quantization
        self.model, self.tokenizer = load(model_id)

    def generate(self, chats: list[list[dict]], params: GenParams) -> list[GenOutput]:
        # mlx_lm (mac-only, always serial -- _batch_size() forces 1 for
        # non-transformers/vLLM backends, plan 005) takes one prompt per call.
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        results = []
        sampler = make_sampler(temp=params.temperature, top_p=params.top_p)
        for messages in chats:
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
            text = mlx_generate(
                self.model, self.tokenizer, prompt,
                max_tokens=params.max_tokens, sampler=sampler, verbose=False,
            )
            results.append(GenOutput(text=_truncate_at_stop(text, params.stop)))
        return results

    def close(self) -> None:
        # mlx arrays are freed by the garbage collector; no explicit handle
        # to release (unlike llama.cpp's ctypes-backed context). Drop our
        # references so the model becomes collectible immediately.
        self.model = None
        self.tokenizer = None
