"""MLXBackend against a fully mocked `mlx_lm` module — mlx_lm is arm64/
darwin-only and cannot be imported on this (linux) CI, and pytest must
never load a real model regardless of platform (Hard rule). Import
injection via sys.modules, same pattern as other backend unit tests.

NOTE (see mlx_be.py module docstring): this only proves the backend calls
mlx_lm's documented API with the right shapes/arguments. It does NOT prove
the backend works on real Apple Silicon — that verification is deferred to
the pre-workshop Mac pass."""

from __future__ import annotations

import sys
import types

import pytest


class FakeTokenizer:
    def __init__(self):
        self.chat_template_calls = []

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        self.chat_template_calls.append({"messages": messages, "tokenize": tokenize})
        return "<prompt>" + messages[-1]["content"]


class FakeModel:
    pass


_LOAD_CALLS = []
_GENERATE_CALLS = []
_SAMPLER_CALLS = []


def _fake_load(path_or_hf_repo, **kwargs):
    _LOAD_CALLS.append({"path_or_hf_repo": path_or_hf_repo, **kwargs})
    return FakeModel(), FakeTokenizer()


def _fake_generate(model, tokenizer, prompt, max_tokens=256, sampler=None, verbose=False, **kwargs):
    _GENERATE_CALLS.append({
        "prompt": prompt, "max_tokens": max_tokens, "sampler": sampler, "verbose": verbose,
    })
    return "ACTION: 2"


def _fake_make_sampler(temp=0.0, top_p=0.0, **kwargs):
    call = {"temp": temp, "top_p": top_p}
    _SAMPLER_CALLS.append(call)
    return call  # opaque marker object is fine; backend only threads it through


@pytest.fixture(autouse=True)
def fake_mlx_lm_module(monkeypatch):
    _LOAD_CALLS.clear()
    _GENERATE_CALLS.clear()
    _SAMPLER_CALLS.clear()

    mlx_lm_mod = types.ModuleType("mlx_lm")
    mlx_lm_mod.load = _fake_load
    mlx_lm_mod.generate = _fake_generate

    sample_utils_mod = types.ModuleType("mlx_lm.sample_utils")
    sample_utils_mod.make_sampler = _fake_make_sampler

    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_mod)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils_mod)
    yield


def _make_backend(model_id="LiquidAI/LFM2.5-VL-1.6B"):
    from slm_rl.inference.mlx_be import MLXBackend

    return MLXBackend(model_id)


def test_load_called_with_model_id():
    _make_backend("LiquidAI/LFM2.5-VL-1.6B")
    assert _LOAD_CALLS[0]["path_or_hf_repo"] == "LiquidAI/LFM2.5-VL-1.6B"


def test_generate_returns_text_from_mlx_lm():
    from slm_rl.inference.base import GenParams

    backend = _make_backend()
    out = backend.generate([[{"role": "user", "content": "go"}]], GenParams(max_tokens=64))
    assert len(out) == 1
    assert out[0].text == "ACTION: 2"


def test_generate_applies_chat_template_per_chat():
    from slm_rl.inference.base import GenParams

    backend = _make_backend()
    chats = [
        [{"role": "user", "content": "a"}],
        [{"role": "user", "content": "b"}],
    ]
    backend.generate(chats, GenParams())
    assert len(backend.tokenizer.chat_template_calls) == 2
    assert _GENERATE_CALLS[0]["prompt"] == "<prompt>a"
    assert _GENERATE_CALLS[1]["prompt"] == "<prompt>b"


def test_generate_threads_sampling_params():
    from slm_rl.inference.base import GenParams

    backend = _make_backend()
    backend.generate([[{"role": "user", "content": "hi"}]], GenParams(max_tokens=99, temperature=0.6, top_p=0.85))
    assert _GENERATE_CALLS[0]["max_tokens"] == 99
    assert _SAMPLER_CALLS[0] == {"temp": 0.6, "top_p": 0.85}


def test_generate_truncates_at_stop_string():
    from slm_rl.inference.base import GenParams

    backend = _make_backend()
    out = backend.generate([[{"role": "user", "content": "hi"}]], GenParams(stop=["ACTION: 2"]))
    assert out[0].text == ""  # stop string is at index 0 -> truncated to empty


def test_close_drops_references():
    backend = _make_backend()
    backend.close()
    assert backend.model is None
    assert backend.tokenizer is None


def test_load_adapter_not_implemented():
    # Documented gap (plan 024): mlx_lm's native LoRA format differs from
    # our PEFT-format adapters; raising here (InferenceBackend default)
    # rather than silently ignoring the adapter is the load-bearing behavior.
    backend = _make_backend()
    with pytest.raises(NotImplementedError):
        backend.load_adapter("some/adapter/path")
