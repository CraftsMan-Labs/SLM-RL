"""Pydantic config models. Plain YAML + these models (no Hydra) — see
docs/DECISIONS.md D6.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

TrainStrategy = Literal["grpo", "reject_sft", "none"]
Backend = Literal["transformers", "transformers-4bit", "mlx", "transformers-vl"]


class MonitorConfig(BaseModel):
    """DoomLoopMonitor thresholds and enabled interventions (D4)."""

    action_repeat_threshold: int = 3
    ngram_loop_threshold: int = 3
    ngram_max_n: int = 4
    state_revisit_threshold: int = 3
    reward_stagnation_window: int = 15
    invalid_streak_threshold: int = 3
    interventions: list[Literal["reflect", "mask_action", "truncate"]] = [
        "reflect",
        "truncate",
    ]
    truncate_penalty: float = -0.5


class GameConfig(BaseModel):
    name: str
    max_turns: int = 100
    invalid_action_penalty: float = -0.2
    retry_penalty: float = -0.05
    shaping_weight: float = 0.1  # auxiliary shaping kept low (reward hygiene)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    eval_episodes: int = 200
    eval_seeds_start: int = 10_000
    extra: dict[str, Any] = Field(default_factory=dict)  # game-specific knobs


class TierConfig(BaseModel):
    """One row of configs/hardware.yaml. First matching tier wins, so order
    tiers from most to least capable."""

    name: str
    model: str
    backend: Backend
    train: TrainStrategy = "none"
    quantization: str | None = "q4"
    # match rules (all present conditions must hold)
    os: Literal["darwin", "linux", "windows"] | None = None
    min_ram_gb: float | None = None
    max_ram_gb: float | None = None
    min_cuda_vram_gb: float | None = None
    requires_mps: bool | None = None


class TrainConfig(BaseModel):
    strategy: TrainStrategy = "reject_sft"
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 5e-5
    # grpo-specific
    group_size: int = 8
    kl_beta: float = 0.02  # KL to previous champion
    # short "ACTION: X" completions sit at ~0.1-0.3 mean token entropy when
    # healthy; measured collapse is ~0.002. 0.05 separates them cleanly.
    entropy_floor: float = 0.05
    entropy_bonus: float = 0.01
    max_completion_tokens: int = 256
    # None / omit = train full epochs. Playground sets 24 for Docker-CPU demos.
    grpo_max_steps: int | None = None
    # Cap exported decision prompts (repo default 512; workshop uses 32).
    grpo_max_prompts: int = 512
    # reject_sft-specific
    episodes_per_generation: int = 200
    selection_quantile: float = 0.25  # keep top-quantile reward trajectories
    exclude_monitor_flagged: bool = True
    max_duplicate_action_sequences: int = 3  # diversity quota vs mode collapse
    # train on rollouts from the last N generations; 1 = current behavior
    replay_generations: int = 3
    win_turn_cap: int = 0  # drop wins longer than this many steps; 0 = disabled
    sft_win_final_dup: int = 1  # write a win's final decision pair this many times; 1 = disabled
    # K episodes per generate() call (transformers/vLLM only; llama.cpp/MLX
    # keep this at 1). 1 = current behavior (serial EpisodeRunner).
    rollout_batch_size: int = 1


class TeacherConfig(BaseModel):
    """Classical-teacher seams (HYBRID_RL.md, D11). Rollout/eval concern, so
    it lives here rather than on TrainConfig; games stay ML-free."""

    pruner: bool = False  # consistent-candidate menu pruning during rollout
    pruner_top_k: int = 10
    pruner_fraction: float = 0.5  # fraction of rollout episodes pruned (rest stay format-mode, matching the gate eval)
    warmstart_episodes: int = 1000  # teacher episodes for the gen-1 warm start
    eval_pruned_episodes: int = 100  # with-pruner side metric; 0 disables; never gated
    # Path to a trained DQN teacher checkpoint (slm-rl train-dqn); None = the
    # game's default teacher (heuristic/solver). A set-but-missing path is an
    # error, never a silent fallback (plan 012).
    dqn_checkpoint: str | None = None


# Prior primary when gen-0 base / imported-SFT eval is skipped (workshop speed).
# Measured boxing SFT champion ~-0.9; -1.9 is a conservative floor so RL gens
# are not gated against a fake 0.0.
DEFAULT_STUB_PRIMARY = -1.9


class GateConfig(BaseModel):
    """EvalGate promotion criteria (D4, training level)."""

    # ~2 sigma on a 500-episode suite; 0.02 blocked a real 0%->1.6% warm start
    min_improvement: float = 0.01  # primary metric margin over champion
    max_intervention_rate_ratio: float = 1.1
    max_invalid_rate: float = 0.05
    min_mean_entropy: float = 0.2
    max_consecutive_failures: int = 2  # then auto-remediate LR + stop evolve loop
    # Used for skipped gen-0 baseline and imported SFT champion stubs.
    stub_primary: float = DEFAULT_STUB_PRIMARY


class RunConfig(BaseModel):
    run_id: str = "default"
    home: str = "./runs"
    game: str = "mastermind"
    generations: int = 2
    seed: int = 0
    tier: str | None = None  # None = auto-detect
    model: str | None = None  # explicit overrides beat the tier table
    backend: Backend | None = None
    train_strategy: TrainStrategy | None = None
    hf_dataset_repo: str | None = None  # push each generation's datasets to this HF dataset repo
    # Workshop bake packs: public HF dataset URL replaces live teacher warm-start.
    # Empty = today's live --warm-start path. Attendees paste; never baked into binary.
    dataset_url: str | None = None
    dqn_url: str | None = None  # optional Atari DQN checkpoint URL/path
    adapter_url: str | None = None  # optional HF model repo with adapter/ (SFT LoRA)
    max_context_tokens: int = 2048  # 8GB budget rule
    train: TrainConfig = Field(default_factory=TrainConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    teacher: TeacherConfig = Field(default_factory=TeacherConfig)
