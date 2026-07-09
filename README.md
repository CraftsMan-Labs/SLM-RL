# SLM-RL

**A self-improving game gymnasium for small language models.**

Small language models learn to play games through reinforcement learning: the model plays games in text-native environments, every decision is collected into a reusable dataset, the model is automatically fine-tuned on its own experience, and the improved model re-enters play. Progress is tracked across *generations* — each promoted only if it demonstrably beats its predecessor.

```
ROLLOUT ──▶ DATASET ──▶ TRAIN ──▶ EVAL ──▶ GATE: promote / rollback
   ▲        (JSONL →   (GRPO or   (frozen         │
   │         parquet)  reject_sft) suites + ELO)  │
   └────────────── champion gen N+1 ◀─────────────┘
```

## Runs on the machine you have

The **entire loop — including training — works on an 8GB RAM machine** with no GPU. The platform detects your hardware and picks the model and training strategy from a config-driven tier table (`configs/hardware.yaml`):

| Your machine | Model | Training strategy |
|---|---|---|
| Any 8GB machine (Mac or CPU laptop) | LFM2.5-350M (q4) | rejection-sampling SFT |
| 16GB Mac | LFM2.5-VL-1.6B | rejection-sampling SFT |
| CUDA GPU 8–16GB | LFM2.5-1.2B | GRPO (TRL + LoRA) |
| CUDA GPU 24GB | gemma-4-E2B-it | GRPO (TRL + LoRA) |

## Launch games (complexity ramp)

1. **Mastermind** — code-breaking, the Phase 1 proof of the full loop
2. **Connect-4** — competitive, self-play league with ELO
3. **Blackjack** — stochastic rewards
4. **Atari Freeway** — via ALE, RAM decoded to text (no vision needed)
5. **Dominion** — deck-building flagship

New games (Catan, driving sims, …) onboard through the plugin contract — see `docs/PLUGIN_GUIDE.md`.

## Anti-doom-loop by design

RL agents get stuck: repeating the same action, revisiting the same states, or collapsing onto a single strategy branch that can't be backtracked out of. SLM-RL counters this at both levels:

- **During play**: a `DoomLoopMonitor` watches every step (action repeats, state revisits, reward stagnation) and escalates — reflect prompt → **backtrack to an earlier state snapshot** → truncate with penalty.
- **During training**: entropy floors with mode-collapse alarms, KL anchoring to the previous champion, and an **EvalGate** — a new generation is only promoted if it beats the champion on a frozen benchmark without regressing on loop/invalid/entropy metrics. Failed generations trigger auto-remediation, optionally including an [antidoom](https://github.com/Liquid4All/antidoom) hygiene stage.

## Status

**Architecture + skeleton.** Interfaces, configs, hardware tier detection, and docs are in place; game engines and trainers land phase by phase (`docs/ROADMAP.md`). What works today:

```bash
uv sync --extra dev
uv run slm-rl info        # detected hardware -> resolved tier -> available games
uv run pytest             # interface & config tests
```

## Docs

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the full system design
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — design decisions D1–D10 with rationale
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — phased build plan
- [`docs/PLUGIN_GUIDE.md`](docs/PLUGIN_GUIDE.md) — how to add a game
