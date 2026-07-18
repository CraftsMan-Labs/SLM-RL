"""GRPO strategy: TRL GRPOTrainer + PEFT LoRA over decision-step prompts with
recorded-return rewards, and an entropy floor that aborts a collapsing run.

Works on CUDA (bf16), Apple Silicon MPS (fp32), and CPU (slow fallback).
All hardware tiers default to this stack via `transformers`; `reject_sft`
remains the teacher / baked-pack warm-start path.

Rewards are dense — every exported decision trains. KL reference: fresh LoRA
-> base model; resumed champion adapter -> TRL snapshots it as a frozen ref.
"""

from __future__ import annotations

import json
from pathlib import Path

from slm_rl.agents.llm_agent import extract_action_token
from slm_rl.datagen.grpo_export import export_grpo_dataset
from slm_rl.training.base import TrainingStrategy, TrainResult
from slm_rl.training.lora import bootstrap_lora, last_log_metrics, release_trainer_memory

try:
    from transformers import TrainerCallback
except ImportError:  # [cuda] extra not installed; create_strategy never gets here
    TrainerCallback = object


def _completion_text(completion) -> str:
    if isinstance(completion, list):
        return str(completion[0].get("content") or "") if completion else ""
    return str(completion)


def _resolved_action(completion, ctx: dict) -> str | None:
    """Map a completion to a legal action id, or None if illegal/unparseable."""
    token = extract_action_token(_completion_text(completion))
    if token is None:
        return None
    legal = [str(a) for a in (ctx.get("legal_actions") or [])]
    if not legal:
        return token  # no menu recorded — accept token as-is for format check
    # index into menu (1-based), same as LLMAgent parse ladder
    if token.isdigit():
        idx = int(token)
        if 1 <= idx <= len(legal):
            return legal[idx - 1]
        return None
    # case-insensitive id / label match against legal ids
    upper = {a.upper(): a for a in legal}
    if token.upper() in upper:
        return upper[token.upper()]
    return None


def format_reward(prompts=None, completions=None, game_ctx=None, **kwargs) -> list[float]:
    """Unparseable -1, parseable-but-illegal -0.5, legal +0.25."""
    out = []
    for completion, ctx_json in zip(completions, game_ctx):
        text = _completion_text(completion)
        if extract_action_token(text) is None:
            out.append(-1.0)
        elif _resolved_action(completion, json.loads(ctx_json)) is None:
            out.append(-0.5)
        else:
            out.append(0.25)
    return out


def return_reward(prompts=None, completions=None, game_ctx=None, **kwargs) -> list[float]:
    """Shape with recorded discounted return when the sample matches a legal
    action. If the sample matches the demonstrator action, use full return;
    other legal actions get a shrunk share of step_reward so groups still
    differentiate without inventing Q-values."""
    out = []
    for completion, ctx_json in zip(completions, game_ctx):
        ctx = json.loads(ctx_json)
        action = _resolved_action(completion, ctx)
        if action is None:
            out.append(0.0)  # format_reward already penalized
            continue
        g = float(ctx.get("discounted_return") or 0.0)
        step_r = float(ctx.get("step_reward") or 0.0)
        target = ctx.get("target_action")
        if target is not None and str(action) == str(target):
            out.append(g)
        else:
            # legal but not the recorded choice — small step signal only
            out.append(0.25 * step_r)
    return out


class EntropyFloorCallback(TrainerCallback):
    """Stop training after `patience` consecutive logs below the floor —
    a collapsed policy must not reach the eval gate as a candidate."""

    def __init__(self, floor: float, patience: int = 3):
        self.floor = floor
        self.patience = patience
        self._streak = 0
        self.collapsed = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        entropy = (logs or {}).get("entropy")
        if entropy is None:
            return
        self._streak = self._streak + 1 if entropy < self.floor else 0
        if self._streak >= self.patience:
            self.collapsed = True
            control.should_training_stop = True


class GRPOStrategy(TrainingStrategy):
    name = "grpo"

    def train(self, dataset_path: Path, out_dir: Path, init_adapter: Path | None = None) -> TrainResult:
        import logging
        import torch
        from datasets import load_dataset
        from trl import GRPOConfig, GRPOTrainer

        logger = logging.getLogger(__name__)

        if self.game_cfg is None:
            raise ValueError("GRPOStrategy needs a GameConfig (env-grounded rewards)")

        out_dir = Path(out_dir)
        grpo_path = Path(dataset_path).parent / "grpo.jsonl"
        n_prompts = export_grpo_dataset(
            dataset_path,
            grpo_path,
            self.game_cfg,
            max_prompts=self.cfg.grpo_max_prompts,
        )
        if n_prompts == 0:
            return TrainResult(adapter_path=init_adapter, metrics={"num_prompts": 0, "skipped": True})

        ds = load_dataset("json", data_files=str(grpo_path), split="train")
        cuda = torch.cuda.is_available()
        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        # bf16 only on CUDA; MPS trains fp32 (fp16 overflows on this stack).
        model, peft_config = bootstrap_lora(self.model_id, self.cfg, init_adapter, cuda)

        # generation_batch_size (= batch * grad_accum) must be divisible by
        # num_generations. Hardware only changes fit; playground caps steps.
        cpu = not (cuda or mps)
        if cuda:
            batch, accum = 8, 2
        elif mps:
            batch, accum = 1, 16
        else:
            batch, accum = 1, 2

        # Docker CPU: group_size 2 keeps generation_batch divisible and wall
        # clock near the ~20 min workshop budget. CUDA/MPS use cfg as-is.
        effective_group_size = min(self.cfg.group_size, 2) if cpu else self.cfg.group_size
        # Ensure batch*accum is divisible by num_generations (TRL requirement).
        if (batch * accum) % effective_group_size != 0:
            accum = effective_group_size  # batch stays 1 on CPU/MPS
        epochs = 1 if cpu else 2
        max_steps = (
            int(self.cfg.grpo_max_steps)
            if self.cfg.grpo_max_steps is not None and self.cfg.grpo_max_steps > 0
            else -1
        )

        # Pre-training memory sanity check
        try:
            import psutil
            avail_gb = psutil.virtual_memory().available / (1024**3)
            logger.info(
                f"GRPO train: available RAM {avail_gb:.1f} GB, "
                f"group_size={effective_group_size}, epochs={epochs}, max_steps={max_steps}"
            )
            if avail_gb < 3.0:
                logger.warning(
                    f"Low memory ({avail_gb:.1f} GB available) — GRPO training may OOM. "
                    "Consider increasing VM memory (OrbStack: Settings > Resources)."
                )
        except ImportError:
            pass

        args = GRPOConfig(
            output_dir=str(out_dir / "trainer"),
            num_generations=effective_group_size,
            temperature=1.3,
            max_completion_length=self.cfg.max_completion_tokens,
            beta=self.cfg.kl_beta,
            entropy_coef=self.cfg.entropy_bonus,
            learning_rate=self.cfg.learning_rate,
            num_train_epochs=epochs,
            max_steps=max_steps,
            per_device_train_batch_size=batch,
            gradient_accumulation_steps=accum,
            bf16=cuda,
            fp16=False,
            use_cpu=not (cuda or mps),
            logging_steps=5,
            save_strategy="no",
            report_to=[],
        )
        watchdog = EntropyFloorCallback(self.cfg.entropy_floor)
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[format_reward, return_reward],
            args=args,
            train_dataset=ds,
            peft_config=peft_config,
            callbacks=[watchdog],
        )
        trainer.train()

        metrics: dict = {
            "num_prompts": n_prompts,
            **last_log_metrics(
                trainer.state.log_history,
                ("reward", "kl", "entropy", "loss", "frac_reward_zero_std"),
            ),
        }

        if watchdog.collapsed:
            metrics["entropy_collapsed"] = True
            adapter_dir = init_adapter
        else:
            adapter_dir = out_dir / "adapter"
            trainer.model.save_pretrained(str(adapter_dir))

        del trainer, model
        release_trainer_memory(cuda)
        if mps:
            torch.mps.empty_cache()
        return TrainResult(adapter_path=adapter_dir, metrics=metrics)
