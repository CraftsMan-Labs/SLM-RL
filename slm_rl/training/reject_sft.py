"""Rejection-sampling SFT (STaR/ReST-style) — the universal training path,
and the gen-0 warm-start for GRPO tiers.

Select winning / top-quantile, monitor-clean trajectories (datagen/
sft_export.py) then LoRA-SFT on the (prompt -> winning action) pairs via TRL
SFTTrainer (completion-only loss + chat template are its defaults)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from slm_rl.datagen.sft_export import export_sft_dataset
from slm_rl.training.base import TrainingStrategy, TrainResult
from slm_rl.training.lora import bootstrap_lora, last_log_metrics, release_trainer_memory


def _metrics_jsonl_callback(metrics_path: Path, num_pairs: int):
    """TRL callback that appends train-step rows for the Evolve monitor UI."""
    from transformers import TrainerCallback

    class _JsonlMetricsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
            if not logs or "loss" not in logs:
                return
            row = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "split": "train",
                "step": int(state.global_step or 0),
                "epoch": float(logs.get("epoch") or state.epoch or 0.0),
                "loss": float(logs["loss"]),
                "learning_rate": logs.get("learning_rate"),
                "num_pairs": num_pairs,
            }
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with metrics_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")

    return _JsonlMetricsCallback()


class RejectSFTStrategy(TrainingStrategy):
    name = "reject_sft"

    def train(self, dataset_path: Path, out_dir: Path, init_adapter: Path | None = None) -> TrainResult:
        import torch
        from datasets import load_dataset
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
        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        # bootstrap_lora's `cuda` flag means "use bf16 weights" — true for CUDA;
        # MPS trains in fp16 via the trainer device placement below.
        model, peft_config = bootstrap_lora(self.model_id, self.cfg, init_adapter, cuda)

        metrics_path = out_dir / "train.metrics.jsonl"
        _append_meta = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "split": "meta",
            "num_pairs": n_pairs,
            "model_id": self.model_id,
        }
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(_append_meta) + "\n")

        # MPS + fp16 overflows on this stack (loss spikes then stays 0 with
        # grad_norm=nan). Train fp32 on Apple Silicon; bf16 only on CUDA.
        args = SFTConfig(
            output_dir=str(out_dir / "trainer"),
            max_length=2048,
            completion_only_loss=True,
            learning_rate=max(self.cfg.learning_rate, 1e-4),
            num_train_epochs=2,
            per_device_train_batch_size=4 if cuda else (1 if mps else 1),
            gradient_accumulation_steps=4 if cuda else (16 if mps else 16),
            bf16=cuda,
            fp16=False,
            use_cpu=not (cuda or mps),
            logging_steps=5,
            save_strategy="no",
            report_to=[],
        )
        trainer = SFTTrainer(model=model, args=args, train_dataset=ds, peft_config=peft_config)
        trainer.add_callback(_metrics_jsonl_callback(metrics_path, n_pairs))
        trainer.train()

        adapter_dir = out_dir / "adapter"
        trainer.model.save_pretrained(str(adapter_dir))

        metrics = {"num_pairs": n_pairs, **last_log_metrics(
            trainer.state.log_history, ("loss", "entropy", "mean_token_accuracy"),
        )}

        del trainer, model
        release_trainer_memory(cuda)
        if mps:
            torch.mps.empty_cache()
        return TrainResult(adapter_path=adapter_dir, metrics=metrics)
