"""GRPO strategy: TRL GRPOTrainer + PEFT LoRA over decision-step prompts with
environment-grounded rewards, and an entropy floor that aborts a collapsing
run (the gate then keeps the champion). Requires the [cuda] extra.

Rewards are dense — every step of every episode trains, no win filter — so
this is the escalation path when reject_sft's win-sparse distillation
plateaus (R7). KL reference: fresh LoRA -> base model; resumed champion
adapter -> TRL snapshots it as a frozen "ref" adapter, i.e. KL to the
previous champion.
"""

from __future__ import annotations

import functools
import gc
import json
import math
from pathlib import Path

from slm_rl.agents.llm_agent import extract_action_token
from slm_rl.datagen.grpo_export import export_grpo_dataset
from slm_rl.games.mastermind.env import score_guess
from slm_rl.teachers.mastermind_solver import consistent_candidates
from slm_rl.training.base import TrainingStrategy, TrainResult
from slm_rl.training.lora import target_modules_for

try:
    from transformers import TrainerCallback
except ImportError:  # [cuda] extra not installed; create_strategy never gets here
    TrainerCallback = object


# -- rewards (pure, CPU-testable) ----------------------------------------
# TRL calls each as f(prompts=, completions=, completion_ids=, **columns);
# with conversational prompts, completions are one-message lists.

def _completion_text(completion) -> str:
    if isinstance(completion, list):
        return str(completion[0].get("content") or "") if completion else ""
    return str(completion)


def _legal_guess(completion, ctx: dict) -> str | None:
    token = extract_action_token(_completion_text(completion))
    if token is None:
        return None
    menu = ctx.get("menu")  # present iff the prompt showed a pruned menu
    if token.isdigit():
        if menu and 1 <= int(token) <= len(menu):
            return menu[int(token) - 1].upper()
        return None
    guess = token.upper()
    ok = (
        len(guess) == len(ctx["secret"])
        and all(c in ctx["colors"] for c in guess)
        and (ctx.get("dup_ok", True) or len(set(guess)) == len(guess))
    )
    if ok and menu and guess not in {m.upper() for m in menu}:
        return None  # off-menu code: rollout's parse_action rejects it too
    return guess if ok else None


def format_reward(prompts=None, completions=None, game_ctx=None, **kwargs) -> list[float]:
    """Unparseable -1, parseable-but-illegal -0.5, legal +0.25."""
    out = []
    for completion, ctx_json in zip(completions, game_ctx):
        text = _completion_text(completion)
        if extract_action_token(text) is None:
            out.append(-1.0)
        elif _legal_guess(completion, json.loads(ctx_json)) is None:
            out.append(-0.5)
        else:
            out.append(0.25)
    return out


@functools.lru_cache(maxsize=1024)  # >= grpo_export's 512-prompt cap
def _consistent_set(colors: str, code_length: int, dup_ok: bool, prior_json: str) -> tuple[str, ...]:
    prior = json.loads(prior_json)
    return tuple(consistent_candidates(colors, code_length, dup_ok, prior))


def deduction_reward(prompts=None, completions=None, game_ctx=None, **kwargs) -> list[float]:
    """Exact elimination reward — HYBRID_RL.md seam 3 with potential
    Phi(s) = -log|consistent(s)|, realized exactly for Mastermind.

    Repeats score -1.0: the old consistency fraction scored a repeated wrong
    guess (k-1)/k — near-max — which is why GRPO never killed the repeat doom
    loop. Otherwise the reward is the fraction of remaining uncertainty the
    guess eliminates (log-scale), so group samples differentiate even when a
    pruned menu makes every option consistent (else reward std hits zero and
    the gradient dies), +1.0 on a direct secret hit.
    """
    out = []
    for completion, ctx_json in zip(completions, game_ctx):
        ctx = json.loads(ctx_json)
        guess = _legal_guess(completion, ctx)
        if guess is None:
            out.append(0.0)  # format_reward already penalized
            continue
        if any(guess == g for g, _, _ in ctx["prior"]):
            out.append(-1.0)  # repeated guess: zero information, doom-loop fuel
            continue
        secret = ctx["secret"]
        before = _consistent_set(
            ctx["colors"], len(secret), ctx.get("dup_ok", True), json.dumps(ctx["prior"])
        )
        r = 0.0
        if len(before) > 1:
            feedback = score_guess(guess, secret)
            n_after = sum(1 for c in before if score_guess(guess, c) == feedback)
            # secret is always in `before` and matches its own feedback -> n_after >= 1
            r = (math.log(len(before)) - math.log(n_after)) / math.log(len(before))
        if guess == secret:
            r += 1.0
        out.append(r)
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
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from trl import GRPOConfig, GRPOTrainer

        if self.game_cfg is None:
            raise ValueError("GRPOStrategy needs a GameConfig (env-grounded rewards)")

        out_dir = Path(out_dir)
        grpo_path = Path(dataset_path).parent / "grpo.jsonl"
        n_prompts = export_grpo_dataset(dataset_path, grpo_path, self.game_cfg)
        if n_prompts == 0:
            return TrainResult(adapter_path=init_adapter, metrics={"num_prompts": 0, "skipped": True})

        ds = load_dataset("json", data_files=str(grpo_path), split="train")
        cuda = torch.cuda.is_available()

        peft_config = None
        model = self.model_id
        if init_adapter is not None:
            from transformers import AutoModelForCausalLM
            from peft import PeftModel

            base = AutoModelForCausalLM.from_pretrained(
                self.model_id, dtype=torch.bfloat16 if cuda else torch.float32
            )
            model = PeftModel.from_pretrained(base, str(init_adapter), is_trainable=True)
        else:
            peft_config = LoraConfig(
                r=self.cfg.lora_rank, lora_alpha=self.cfg.lora_alpha, lora_dropout=0.05,
                bias="none", task_type="CAUSAL_LM",
                target_modules=target_modules_for(self.model_id),
            )

        # generation_batch_size (= batch * grad_accum on one device) must be
        # divisible by num_generations: 8*2 and 2*8 both work for group<=16.
        args = GRPOConfig(
            output_dir=str(out_dir / "trainer"),
            num_generations=self.cfg.group_size,
            # frac_reward_zero_std reached 1.0 at the default temp 1.0 (all 8
            # samples identical -> zero advantage); hotter sampling keeps groups diverse
            temperature=1.3,
            max_completion_length=self.cfg.max_completion_tokens,
            beta=self.cfg.kl_beta,
            entropy_coef=self.cfg.entropy_bonus,
            learning_rate=self.cfg.learning_rate,
            num_train_epochs=1,
            per_device_train_batch_size=8 if cuda else 2,
            gradient_accumulation_steps=2 if cuda else 8,
            bf16=cuda,
            use_cpu=not cuda,
            logging_steps=5,
            save_strategy="no",
            report_to=[],
        )
        watchdog = EntropyFloorCallback(self.cfg.entropy_floor)
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[format_reward, deduction_reward],
            args=args,
            train_dataset=ds,
            peft_config=peft_config,
            callbacks=[watchdog],
        )
        trainer.train()

        metrics: dict = {"num_prompts": n_prompts}
        # frac_reward_zero_std ~1.0 means dead gradient (all group samples
        # scored identically) — surfaced so acceptance runs can verify
        for key in ("reward", "kl", "entropy", "loss", "frac_reward_zero_std"):
            vals = [row[key] for row in trainer.state.log_history if key in row]
            if vals:
                metrics[key] = vals[-1]

        if watchdog.collapsed:
            metrics["entropy_collapsed"] = True
            adapter_dir = init_adapter  # discard the collapsed policy
        else:
            adapter_dir = out_dir / "adapter"
            trainer.model.save_pretrained(str(adapter_dir))

        del trainer, model
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
        return TrainResult(adapter_path=adapter_dir, metrics=metrics)
