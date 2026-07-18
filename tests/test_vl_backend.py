"""VLTransformersBackend against fully mocked `torch`/`transformers`/`peft`
modules — Hard rule: pytest never loads a real model or needs the [cuda]
extra installed. Import-injection via sys.modules, same pattern as
test_mlx_backend.py."""

from __future__ import annotations

import sys
import types

import pytest


class FakeTensor1D:
    """Stands in for a 1D torch.Tensor slice/mean -- one batch row of
    compute_transition_scores' real (batch, gen_len) output."""

    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return FakeTensor1D(self.values[item])
        return self.values[item]

    def mean(self):
        return FakeScalar(sum(self.values) / len(self.values) if self.values else 0.0)


class FakeTensor:
    """Stands in for compute_transition_scores' (batch, gen_len) 2D tensor
    -- `scores[i]` (int index) returns a 1D row, matching real tensor
    semantics (contrast a plain list, where `values[0]` would be a scalar)."""

    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return FakeTensor(self.values[item])
        return FakeTensor1D(self.values[item])

    @property
    def shape(self):
        return (1, len(self.values))


class FakeScalar:
    def __init__(self, v):
        self.v = v

    def __float__(self):
        return float(self.v)


class FakeBatchInputs(dict):
    """Stands in for the dict returned by processor.apply_chat_template
    (return_dict=True) -- a dict of tensors with a `.to(device)` method."""

    def to(self, device):
        self.device_seen = device
        return self


class FakeInputIds:
    def __init__(self, n):
        self._n = n

    @property
    def shape(self):
        return (1, self._n)


class FakeSequences:
    """out.sequences[:, prompt_len:] -> the generated ids slice."""

    def __init__(self, full_len, gen_ids):
        self.full_len = full_len
        self.gen_ids = gen_ids

    def __getitem__(self, item):
        # item is (slice(None), slice(prompt_len, None))
        return FakeGenIds(self.gen_ids)


class FakeGenIds:
    def __init__(self, ids):
        self.ids = ids

    @property
    def shape(self):
        return (1, len(self.ids))


class FakeGenerateOutput:
    def __init__(self, gen_ids):
        self.sequences = FakeSequences(full_len=5 + len(gen_ids), gen_ids=gen_ids)
        self.scores = ["score"] * len(gen_ids)


_GENERATE_CALLS = []
_FROM_PRETRAINED_CALLS = []
_CHAT_TEMPLATE_CALLS = []
_GEN_IDS = [101, 102, 103]  # arbitrary token ids the fake model "generates"


class FakeModel:
    device = "cpu"

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        _GENERATE_CALLS.append(kwargs)
        return FakeGenerateOutput(_GEN_IDS)

    def compute_transition_scores(self, sequences, scores, normalize_logits=True):
        # (batch=1, gen_len) -- one row of per-token logprobs.
        return FakeTensor([[-0.1, -0.2, -0.3][: len(_GEN_IDS)]])


class FakeAutoModelForImageTextToText:
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        _FROM_PRETRAINED_CALLS.append({"model_id": model_id, **kwargs})
        return FakeModel()


class FakeProcessor:
    def __init__(self, model_id):
        self.model_id = model_id

    def apply_chat_template(self, messages, **kwargs):
        _CHAT_TEMPLATE_CALLS.append({"messages": messages, **kwargs})
        return FakeBatchInputs(input_ids=FakeInputIds(5))

    def batch_decode(self, gen_ids, skip_special_tokens=True):
        return ["ACTION: 2"]


class FakeAutoProcessor:
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return FakeProcessor(model_id)


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeTorch(types.ModuleType):
    bfloat16 = "bfloat16"
    float32 = "float32"

    class cuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def empty_cache():
            pass

    class backends:
        mps = None

    @staticmethod
    def inference_mode():
        return _Ctx()


@pytest.fixture(autouse=True)
def fake_ml_modules(monkeypatch):
    _GENERATE_CALLS.clear()
    _FROM_PRETRAINED_CALLS.clear()
    _CHAT_TEMPLATE_CALLS.clear()

    fake_torch = FakeTorch("torch")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForImageTextToText = FakeAutoModelForImageTextToText
    fake_transformers.AutoProcessor = FakeAutoProcessor
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    yield


def _make_backend(model_id="LiquidAI/LFM2.5-VL-450M"):
    from slm_rl.inference.transformers_vl_be import VLTransformersBackend

    return VLTransformersBackend(model_id)


def test_default_model_id_is_450m():
    from slm_rl.inference.transformers_vl_be import DEFAULT_VL_MODEL

    assert DEFAULT_VL_MODEL == "LiquidAI/LFM2.5-VL-450M"


def test_loads_via_automodel_for_image_text_to_text_and_autoprocessor():
    _make_backend("LiquidAI/LFM2.5-VL-1.6B")
    assert _FROM_PRETRAINED_CALLS[0]["model_id"] == "LiquidAI/LFM2.5-VL-1.6B"


def test_factory_resolves_transformers_vl():
    from slm_rl.inference.base import create_backend

    be = create_backend("transformers-vl", "LiquidAI/LFM2.5-VL-450M")
    from slm_rl.inference.transformers_vl_be import VLTransformersBackend

    assert isinstance(be, VLTransformersBackend)


def test_generate_reaches_processor_with_image_content_part():
    from slm_rl.inference.base import GenParams

    backend = _make_backend()
    image = object()  # opaque marker: backend never inspects image internals
    chats = [[
        {"role": "system", "content": [{"type": "text", "text": "sys"}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "go"},
        ]},
    ]]
    out = backend.generate(chats, GenParams(max_tokens=64))
    assert len(out) == 1
    call = _CHAT_TEMPLATE_CALLS[0]
    parts = call["messages"][1]["content"]
    assert any(p.get("type") == "image" and p.get("image") is image for p in parts)
    assert call["add_generation_prompt"] is True
    assert call["tokenize"] is True


def test_generate_returns_text_and_mean_logprob():
    from slm_rl.inference.base import GenParams

    backend = _make_backend()
    chats = [[{"role": "user", "content": [{"type": "text", "text": "go"}]}]]
    out = backend.generate(chats, GenParams())
    assert out[0].text == "ACTION: 2"
    assert out[0].logprob == pytest.approx((-0.1 + -0.2 + -0.3) / 3)


def test_generate_threads_sampling_params():
    from slm_rl.inference.base import GenParams

    backend = _make_backend()
    chats = [[{"role": "user", "content": [{"type": "text", "text": "go"}]}]]
    backend.generate(chats, GenParams(max_tokens=77, temperature=0.6, top_p=0.85))
    call = _GENERATE_CALLS[0]
    assert call["max_new_tokens"] == 77
    assert call["temperature"] == 0.6
    assert call["top_p"] == 0.85
    assert call["do_sample"] is True


def test_greedy_when_temperature_zero():
    from slm_rl.inference.base import GenParams

    backend = _make_backend()
    chats = [[{"role": "user", "content": [{"type": "text", "text": "go"}]}]]
    backend.generate(chats, GenParams(temperature=0.0))
    assert _GENERATE_CALLS[0]["do_sample"] is False
    assert _GENERATE_CALLS[0]["temperature"] is None


def test_generate_truncates_at_stop_string():
    from slm_rl.inference.base import GenParams

    backend = _make_backend()
    chats = [[{"role": "user", "content": [{"type": "text", "text": "go"}]}]]
    out = backend.generate(chats, GenParams(stop=["ACTION"]))
    assert out[0].text == ""  # "ACTION" is at index 0 of "ACTION: 2" -> truncated to empty


def test_close_frees_model():
    backend = _make_backend()
    backend.close()
    assert not hasattr(backend, "model")


def test_load_adapter_wraps_with_peft(monkeypatch):
    class FakePeftModel:
        instances = []

        @classmethod
        def from_pretrained(cls, model, adapter_path, adapter_name):
            inst = cls()
            inst.model = model
            inst.adapter_path = adapter_path
            inst.adapter_name = adapter_name
            cls.instances.append(inst)
            return inst

    fake_peft = types.ModuleType("peft")
    fake_peft.PeftModel = FakePeftModel
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    backend = _make_backend()
    backend.load_adapter("some/adapter/path")
    assert isinstance(backend.model, FakePeftModel)
    assert backend.model.adapter_path == "some/adapter/path"
