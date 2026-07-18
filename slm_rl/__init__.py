"""SLM-RL: a self-improving game gymnasium for small language models.

The generation loop:

    ROLLOUT -> DATASET -> TRAIN (grpo | reject_sft) -> EVAL -> GATE -> promote/rollback

Core rule: every layer has an 8GB-RAM path. Heavy dependencies (torch, trl,
vllm, mlx, ale-py, openenv) are optional extras and must only be imported
lazily inside the modules that need them.
"""

__version__ = "0.1.0"
