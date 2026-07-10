"""Rejection-sampling SFT (STaR/ReST-style) — the universal training path,
and the gen-0 warm-start for GRPO tiers.

Select winning / top-quantile, monitor-clean trajectories (datagen/
sft_export.py) then LoRA-SFT on the (prompt -> winning action) pairs via TRL
SFTTrainer (completion-only loss + chat template are its defaults)."""

from __future__ import annotations

import gc
from pathlib import Path

from slm_rl.datagen.sft_export import export_sft_dataset
from slm_rl.training.base import TrainingStrategy, TrainResult
from slm_rl.training.lora import target_modules_for


class RejectSFTStrategy(TrainingStrategy):
    name = "reject_sft"

    def train(self, dataset_path: Path, out_dir: Path, init_adapter: Path | None = None) -> TrainResult:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from trl import SFTConfig, SFTTrainer

        out_dir = Path(out_dir)
        sft_path = Path(dataset_path).parent / "sft.jsonl"
        n_pairs = export_sft_dataset(dataset_path, sft_path, self.cfg)
        if n_pairs == 0:
            # Nothing worth training on this generation — return the champion
            # unchanged; the gate will see no improvement and reject.
            return TrainResult(adapter_path=init_adapter, metrics={"num_pairs": 0, "skipped": True})

        ds = load_dataset("json", data_files=str(sft_path), split="train")
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

        args = SFTConfig(
            output_dir=str(out_dir / "trainer"),
            max_length=2048,
            completion_only_loss=True,
            learning_rate=max(self.cfg.learning_rate, 1e-4),
            num_train_epochs=2,
            per_device_train_batch_size=4 if cuda else 1,
            gradient_accumulation_steps=4 if cuda else 16,
            bf16=cuda,
            use_cpu=not cuda,
            logging_steps=5,
            save_strategy="no",
            report_to=[],
        )
        trainer = SFTTrainer(model=model, args=args, train_dataset=ds, peft_config=peft_config)
        trainer.train()

        adapter_dir = out_dir / "adapter"
        trainer.model.save_pretrained(str(adapter_dir))

        logs = trainer.state.log_history
        metrics = {"num_pairs": n_pairs}
        for key in ("loss", "entropy", "mean_token_accuracy"):
            vals = [row[key] for row in logs if key in row]
            if vals:
                metrics[key] = vals[-1]

        del trainer, model
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
        return TrainResult(adapter_path=adapter_dir, metrics=metrics)
