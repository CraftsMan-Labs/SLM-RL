"""CPU-first causal-LM load avoids hanging MPS from_pretrained."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from slm_rl.training import lora as lora_mod


def test_load_causal_lm_cpu_first_then_mps():
    fake = MagicMock(name="model")
    fake.to.return_value = fake
    torch = MagicMock()
    torch.bfloat16 = "bf16"
    torch.float32 = "fp32"
    torch.backends.mps.is_available.return_value = True
    torch.get_num_threads.return_value = 8
    auto = MagicMock()
    auto.from_pretrained.return_value = fake
    transformers = MagicMock()
    transformers.AutoModelForCausalLM = auto

    with patch.dict("sys.modules", {"torch": torch, "transformers": transformers}):
        out = lora_mod.load_causal_lm("LiquidAI/LFM2.5-1.2B-Instruct", cuda=False)

    auto.from_pretrained.assert_called_once()
    assert auto.from_pretrained.call_args.kwargs["low_cpu_mem_usage"] is True
    assert auto.from_pretrained.call_args.kwargs["dtype"] == "fp32"
    fake.to.assert_called_once_with("mps")
    torch.set_num_threads.assert_any_call(1)
    torch.set_num_threads.assert_any_call(8)
    assert out is fake


def test_bootstrap_lora_returns_model_not_id_string():
    fake = MagicMock(name="model")
    cfg = SimpleNamespace(lora_rank=8, lora_alpha=16)
    peft = MagicMock()
    peft.LoraConfig.return_value = MagicMock(name="lora_cfg")

    with (
        patch.dict("sys.modules", {"peft": peft}),
        patch.object(lora_mod, "load_causal_lm", return_value=fake),
    ):
        model, peft_config = lora_mod.bootstrap_lora(
            "LiquidAI/LFM2.5-1.2B-Instruct", cfg, None, cuda=False
        )

    assert model is fake
    assert peft_config is not None
    assert not isinstance(model, str)
