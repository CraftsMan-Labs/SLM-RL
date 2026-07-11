# SLM-RL Architecture

A self-improving game gymnasium for small language models: play → dataset → fine-tune → eval gate → promote → play again.

> Visual walkthrough: [PIPELINE.md](PIPELINE.md) — the 0→1 generation
> pipeline as a Mermaid diagram, with the hard-won rules encoded in it.

## The generation loop

The system is organized around a **generation loop** run by the orchestrator (`slm_rl/orchestrator/generation.py`). One *generation* = one full cycle:

```
                ┌──────────────────────────────────────────────────────┐
                │                 GenerationRunner                     │
                └──────────────────────────────────────────────────────┘
  ┌───────────┐   ┌────────────┐   ┌───────────┐   ┌──────────┐   ┌─────────────┐
  │  ROLLOUT  │──▶│  DATASET   │──▶│   TRAIN   │──▶│   EVAL   │──▶│ GATE:       │
  │ agent ×   │   │ JSONL →    │   │ grpo or   │   │ frozen   │   │ promote or  │
  │ games ×   │   │ parquet    │   │ reject_sft│   │ suites + │   │ rollback    │
  │ monitors  │   │            │   │ (+LoRA)   │   │ ELO      │   │             │
  └───────────┘   └────────────┘   └───────────┘   └──────────┘   └─────────────┘
        ▲                                                               │
        └───────────────── champion model (gen N+1) ◀───────────────────┘
```

## Five decoupled layers

1. **Games** (`slm_rl/games/`) — pure-Python, seed-deterministic, text-native engines implementing one `Game` ABC (`games/base.py`). No ML dependencies. Registered via `@register_game` + setuptools entry points (plugin system).
2. **Agents & Inference** (`slm_rl/agents/`, `slm_rl/inference/`) — an `Agent` turns observations into `ActionDecision`s via a pluggable `InferenceBackend` (transformers CUDA/MPS, vLLM, llama.cpp GGUF, MLX). Hardware tier detection (`platform/hardware.py`) selects model + backend from `configs/hardware.yaml`.
3. **Rollout & Datagen** (`slm_rl/rollout/`, `slm_rl/datagen/`) — `EpisodeRunner` with the `DoomLoopMonitor` wired in; every decision persisted as a `RolloutRecord` (streamed JSONL → consolidated parquet). Datasets are a first-class product with a versioned schema (`datagen/schema.py`), reusable for SFT warm-start and offline RL.
4. **Training** (`slm_rl/training/`) — two `TrainingStrategy` implementations behind one interface: `grpo` (TRL + PEFT LoRA, CUDA) and `reject_sft` (rejection-sampling SFT, universal). Optional antidoom hygiene stage.
5. **Eval & Orchestration** (`slm_rl/eval/`, `slm_rl/orchestrator/`) — fixed-seed benchmark suites, ELO league with bot anchors, `EvalGate` promotion/rollback, on-disk `ModelRegistry`, metrics + streamlit dashboard.
6. **Teachers** (`slm_rl/teachers/` — see HYBRID_RL.md / D11) — cheap classical policies (exact solvers today, CleanRL-pattern DQN over `Game.vector_obs()` planned) that implement the same `Agent` ABC. They feed the loop at three seams — trajectory warm-start via `reject_sft` (`evolve --warm-start`), top-k action-menu pruning during rollout (`--pruner`), and potential-based reward shaping during GRPO (Mastermind: the exact elimination reward) — and are barred from eval, so the gate always measures pure LLM skill.

## The 8GB principle

**Every layer has an 8GB-RAM path** — the full self-improvement loop runs on an 8GB machine with no GPU. Enforced budget rules:

- Default 8GB model: `LFM2.5-350M` (text) / `LFM2.5-VL-450M` (vision), 4-bit quantized (≤1GB resident) via llama.cpp or MLX — never vLLM or fp16 transformers on this tier.
- **One model resident at a time**: frozen-generation opponents are LoRA adapter hot-swaps, never a second model copy.
- Context capped (≤2048 tokens); observation renderers are designed to stay small.
- Rollouts stream to JSONL (never accumulated in RAM); parquet consolidation is chunked.
- ALE runs in-process via `ale-py` (~100MB), not the Docker OpenEnv image, on low-RAM tiers.
- Heavy dependencies (torch, trl, vllm, mlx, ale-py, openenv) are optional extras, imported lazily — the core package imports clean with none installed.
- CI gate: a mini full loop (rollout → consolidate → reject_sft → eval) must complete under a hard 7GB memory cap.

Training is **tier-adaptive** (see DECISIONS.md D10): CUDA machines use GRPO for fastest improvement; everything else uses rejection-sampling SFT — same rollout runner, dataset schema, eval gate, and registry, only the TRAIN box differs.

## External ecosystem positioning

- **Gymnasium-style core, OpenEnv at the edge**: games implement our own text-native contract; `bridges/gym_adapter.py` wraps external Gymnasium envs (ALE) in, and `bridges/openenv_bridge.py` wraps our games out as OpenEnv servers (TRL `environment_factory` path, Phase 5). OpenEnv is experimental/breaking, so it's an optional pinned extra — and in-process rollouts are 10–100× cheaper per step than HTTP.
- **antidoom** (Liquid AI) inspired the doom-loop concepts; the actual FTPO tool is an optional between-generations hygiene stage (`training/hygiene.py`).
- The user-shared OpenEnv reference (`examples/atari_simple.py` upstream) is recreated against our API in `examples/atari_simple.py` — same reset/step/legal_actions shape, no Docker required.

## On-disk layout

```
runs/<run_id>/
  run_config.yaml              frozen resolved config
  registry.json                champion pointer + history + ELO
  generations/gen_NNN/
    adapter/                   PEFT LoRA adapter (unmerged, hot-swappable)
    rollouts/*.jsonl           raw per-episode step records
    dataset/train.parquet      consolidated training view
    eval/results.json
    metrics.json               train + eval + doom-loop stats
    MANIFEST.json              base model id, parent gen, config hash, git sha
datasets/                      cross-run consolidated parquet (the data product)
```

Adapters stay unmerged; `slm-rl export --gen N --merge [--gguf]` produces merged/quantized artifacts for Mac play.

## Live-play viewer (`slm-rl watch`)

`slm_rl/webui/` is a browser view of a run in progress: `slm-rl watch --run
<id>` tails `runs/<run_id>/generations/gen_*/rollouts/*.jsonl` (the same
`RolloutRecord` stream datagen already writes) and pushes each decision to
the browser over Server-Sent Events as colored-peg episode cards — guess,
reward, parse status, monitor flags, and the raw completion. It is a pure
*tail, parse, push* layer: no hooks into the runner, no writes into `runs/`
(CODING_GUIDELINE invariant 5 — read-only observers stay read-only), and no
new dependency of any kind. Like core `slm_rl`, it is stdlib-only
(`http.server`, `json`, `threading`) so it holds on the 8GB tier with none
of the optional extras installed; it is a different surface from the
Streamlit `[dashboard]` metrics stub in `slm_rl/dashboard/`, which stays a
separate, heavier, cross-generation curve viewer.

For Atari games, each episode card also gets a "▶ watch" button (plan 010)
that streams the actual game screen from `GET /frames?episode=<id>`. No
frames are recorded during rollout — plan 008 established that ALE with
`repeat_action_probability=0.0` is byte-deterministic given a seed and
action script, and every record already carries `(seed, parsed_action)` per
step, so `slm_rl/webui/replay.py` re-simulates the episode in a fresh env
and renders real frames on demand (a minimal stdlib PNG encoder,
`slm_rl/webui/png.py`, avoids a Pillow dependency). This stays a read-only
observer: replay never touches the rollout/training path, and non-Atari
games (Mastermind) get a 501 by design.

## Workshop playground (`slm-rl playground`)

`slm_rl/playground/` is a small workshop UI (plan 013) for attendees to
tweak the system and *measure* whether their change helped, without
touching the repo: a knob form (max_turns, reward-shaping constants,
monitor thresholds, selection quantile, teacher choice, ...) plus a reward-
code tab, a "run experiment" button that launches a quick CPU screen
(`--agent solver|random`, ~1-3 min), and a scoreboard comparing mean/
median/max score, action mix, and intervention counts across experiments.
Like `slm_rl/webui/`, it is stdlib-only (`http.server`, `json`,
`threading`), so it holds on the 8GB tier with no optional extras
installed — but unlike the viewer, it is a read-**write** surface: it
writes only under `runs/playground/<name>/`, never elsewhere.

**Reproducibility by construction.** Each experiment is materialized to
`runs/playground/<name>/config/{default.yaml, games/<game>.yaml}` — the
repo configs deep-merged with the attendee's knob values — then a `rollout`
(or `evolve`) subprocess is launched pointed at that directory via
`--config-dir` (both config loaders already accepted this; the CLI just
didn't expose it). The experiment directory IS its config, and every
experiment gets its own run-id (`pg-<name>`), so playground numbers are
never silently compared against a main run.

**The reward-code seam.** The gym adapter gained an optional
`extra["reward_hook"]`: a path to a Python file defining `shape_reward(ctx)
-> float`, loaded via `importlib` and applied as a thin wrapper around the
existing reward formula. Absent, the code path is byte-identical to before
this knob existed (proven by `tests/test_reward_hook.py`). This reshapes
*training-time* reward — which drives reject_sft's top-quantile episode
selection, the pedagogical point of the exercise — but the eval gate's
primary metric is mean raw score parsed from `outcome`, which the hook
cannot touch (CODING_GUIDELINE invariant 2: eval-gate purity). Monitor-side
penalties (retry/fallback/truncate) are also outside the hook's reach — they
are applied later, in the rollout runner.

**Trust model.** Executing attendee-written Python is by design: this is a
local tool run on the attendee's own machine, the same trust model as them
editing the repo directly. It binds `127.0.0.1` by default and should not be
exposed beyond localhost.

## Anti-doom-loop design (two levels)

**Rollout level — `DoomLoopMonitor`** (`rollout/monitor.py`), per step:

| Signal | Definition | Default |
|---|---|---|
| `action_repeat` | identical-action streak | ≥3 |
| `ngram_loop` | action n-gram (n=2–4) repeated consecutively | ≥3× |
| `state_revisit` | visits to current `state_hash()` in episode | ≥3 |
| `reward_stagnation` | steps since cumulative reward last rose | ~15 (per game) |
| `invalid_streak` | consecutive invalid actions | ≥3 |

Intervention ladder (escalating, all logged into `RolloutRecord.monitor_flags`):
1. **Reflect** — inject a nudge listing untried moves + mask the looping action from the menu for one turn.
2. **Backtrack** — restore an earlier `Game.snapshot()` checkpoint and mark the looping branch (the explicit answer to "the system needs a way to backtrack").
3. **Truncate** — end the episode with a shaped penalty (−0.5), `truncated_by_monitor=True`.

**Training level** (prevents reward concentrating on one branch):
- Entropy floor + bonus, with abort-and-rollback on sustained collapse.
- KL to the *previous champion* (bounded drift, compounding progress).
- Reward hygiene: clipped shaping (low weight ~0.1–0.2), programmatic rule-based rewards only.
- **EvalGate**: promote only if primary metric ≥ champion + margin AND intervention rate, invalid rate, and entropy don't regress. Two consecutive failures → auto-remediation (halve LR, raise entropy bonus, optional antidoom stage). Rollback = the champion pointer never moves.

## Verification strategy

- Per-game engines: pure-Python unit tests (legal moves, terminal detection, scoring) — CPU, no models.
- Parsing ladder: golden tests over malformed completions.
- End-to-end smoke (CPU): `slm-rl rollout --game mastermind --agent random` → valid JSONL → parquet.
- Full-loop proof: `slm-rl evolve --game mastermind --generations 5` with a rising win-rate curve, on both an 8GB machine (reject_sft) and a CUDA card (GRPO).
- 8GB memory-budget CI gate (hard cap, peak RSS < 7GB).
- Doom-loop machinery: a scripted degenerate agent must trigger reflect → backtrack → truncate in order.
