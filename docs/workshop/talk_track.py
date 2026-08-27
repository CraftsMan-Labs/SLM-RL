"""Shared Colab ↔ presentation talk track.

The notebook and the Vue deck both import this mapping so chapter headings,
slide-range cues, and section dividers stay aligned. Keep it free of torch
and slm_rl imports.
"""

from __future__ import annotations

from typing import Any

# Presentation slide ids that open each Colab chapter. Not every slide is
# mirrored — only the cue that tells the instructor where to flip the deck.
CHAPTERS: list[dict[str, Any]] = [
    {
        "number": 0,
        "title": "Setup",
        "heading": "## 0. Setup",
        "goal": "Land on a T4, install the stack, and lock the workshop knobs.",
        "slides": "cover → join-lobby → why-slm-matters → today",
        "slide_ids": ("cover", "join-lobby", "why-slm-matters", "today"),
        "diagram": "evolve-loop",
    },
    {
        "number": 1,
        "title": "The games",
        "heading": "## 1. The games",
        "goal": "See Atari as text: RAM in, a numbered action menu out.",
        "slides": "what-is-slm → why-games → quiz-1",
        "slide_ids": ("section-games", "what-is-slm", "why-games", "quiz-1", "quiz-1-reveal"),
        "diagram": "games-pipeline",
    },
    {
        "number": 2,
        "title": "Config",
        "heading": "## 2. Config",
        "goal": "Watch default.yaml, the game YAML, and form overrides merge.",
        "slides": "journey-tech (config lives in the notebook; deck stays on the story)",
        "slide_ids": ("section-config",),
        "diagram": "config-merge",
    },
    {
        "number": 3,
        "title": "The model plays",
        "heading": "## 3. The model plays",
        "goal": "One observation becomes a prompt, then a parsed ActionSpec.",
        "slides": "journey → journey-tech",
        "slide_ids": ("section-model", "journey", "journey-tech"),
        "diagram": "parse-action",
    },
    {
        "number": 4,
        "title": "Rollout + dataset",
        "heading": "## 4. Rollout + dataset",
        "goal": "Write one JSONL episode and inspect the training schema.",
        "slides": "rl-loop → sft-vs-rl → why-sft-first",
        "slide_ids": ("section-data", "rl-loop", "sft-vs-rl", "why-sft-first"),
        "diagram": "rollout-dataset",
    },
    {
        "number": 5,
        "title": "Teachers / hybrid RL",
        "heading": "## 5. Teachers / hybrid RL",
        "goal": "Train a mute DQN, then compare it with the SLM. Mario is optional intuition.",
        "slides": "dqn → what-is-dqn → mechanism slides → dqn-mario → teacher-dataset",
        "slide_ids": (
            "section-dqn",
            "dqn",
            "what-is-dqn",
            "dqn-q-values",
            "dqn-bellman",
            "dqn-replay",
            "dqn-target",
            "dqn-epsilon",
            "dqn-curve",
            "dqn-mario",
            "dqn-bridge",
            "teacher-dataset",
        ),
        "diagram": "dqn-hybrid",
    },
    {
        "number": 6,
        "title": "Packs",
        "heading": "## 6. Packs",
        "goal": "Bundle teacher demos and the DQN checkpoint so the room shares homework.",
        "slides": "workshop-flow (DQN teacher → ~20 demos)",
        "slide_ids": ("section-packs",),
        "diagram": "packs",
    },
    {
        "number": 7,
        "title": "Training",
        "heading": "## 7. Training",
        "goal": "Choose reject_sft, GRPO, or both, then write a LoRA adapter.",
        "slides": "gen-0-1 → demo-sft → rlvr → grpo → grpo-analogy → grpo-tech",
        "slide_ids": (
            "section-train",
            "gen-0-1",
            "demo-sft",
            "rlvr",
            "grpo",
            "grpo-analogy",
            "grpo-tech",
            "stack",
        ),
        "diagram": "train-strategies",
    },
    {
        "number": 8,
        "title": "Eval and the gate",
        "heading": "## 8. Eval and the gate",
        "goal": "Promote only when a frozen exam says the SLM itself got better.",
        "slides": "promote-reject → eval-gate",
        "slide_ids": ("section-gate", "promote-reject", "eval-gate"),
        "diagram": "eval-gate",
    },
    {
        "number": 9,
        "title": "The evolve loop",
        "heading": "## 9. The evolve loop",
        "goal": "Run one generation: rollout, train, exam, promote or keep the champion.",
        "slides": "self-improve → champion-rollouts",
        "slide_ids": ("section-evolve", "self-improve", "champion-rollouts"),
        "diagram": "evolve-loop",
    },
    {
        "number": 10,
        "title": "Theater",
        "heading": "## 10. Theater",
        "goal": "Replay base vs champion on the same seeds, side by side.",
        "slides": "improvement → workshop-flow",
        "slide_ids": ("section-theater", "improvement", "workshop-flow"),
        "diagram": "theater",
    },
    {
        "number": 11,
        "title": "Publish",
        "heading": "## 11. Publish",
        "goal": "Opt in to push the run to Hugging Face. A missing token is a no-op.",
        "slides": "closing (publish is notebook-only)",
        "slide_ids": ("section-publish",),
        "diagram": "publish",
    },
    {
        "number": 12,
        "title": "Build your own game",
        "heading": "## 12. Build your own game",
        "goal": "Register a tiny Game ABC and roll it out with the same runner.",
        "slides": "beyond-atari",
        "slide_ids": ("section-plugin", "beyond-atari"),
        "diagram": "game-abc",
    },
    {
        "number": 13,
        "title": "Tests",
        "heading": "## 13. Tests",
        "goal": "Run a fast CPU slice, then close.",
        "slides": "reward-hacking → quiz-2 → closing",
        "slide_ids": (
            "reward-hacking",
            "reward-imbalance",
            "doom-loops",
            "quiz-2",
            "closing",
        ),
        "diagram": None,
    },
]


def chapter_by_number(number: int) -> dict[str, Any]:
    for row in CHAPTERS:
        if row["number"] == number:
            return row
    raise KeyError(number)


def chapter_for_slide(slide_id: str) -> dict[str, Any] | None:
    for row in CHAPTERS:
        if slide_id in row["slide_ids"]:
            return row
    return None


def colab_cue(number: int) -> str:
    row = chapter_by_number(number)
    return f"Presentation: {row['slides']}"


def presentation_cue(slide_id: str) -> str:
    row = chapter_for_slide(slide_id)
    if row is None:
        return ""
    return f"Colab · {row['number']}. {row['title']}"
