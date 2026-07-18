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
| Any 8GB machine (Mac or CPU laptop) | LFM2.5-350M (transformers) | GRPO (TRL + LoRA) |
| 16GB Mac | LFM2.5-1.2B-Instruct (transformers / MPS) | GRPO (TRL + LoRA) |
| CUDA GPU 8–16GB | LFM2.5-1.2B-Instruct | GRPO (TRL + LoRA) |
| CUDA GPU 24GB | gemma-4-E2B-it | GRPO (TRL + LoRA) |

Bare-metal installs are extra-gated so the 8GB floor never pulls CUDA wheels
it doesn't need:

- `uv sync --extra cpu-train --extra atari --extra dev` — CPU torch +
  transformers/trl/peft (PyTorch CPU wheel index) for evolve on the floor tier.
- `uv sync --extra atari --extra mac --extra dev` — Apple Silicon extras
  (MLX optional via `--backend mlx`; see `slm_rl/inference/mlx_be.py`).
- `uv sync --extra cuda --extra atari --extra dev` — CUDA GRPO stack.

Default tiers use `transformers`, so multi-generation evolve loads PEFT
adapters natively. MLX remains an optional inference override.

## Launch games (workshop slate)

Four Atari keepers. Each ships with a DQN (or heuristic) teacher, bake pack,
reject_sft warm-start, and CUDA/MPS GRPO:

1. **Boxing** — ALE RAM→text; signed punch score (`mean_score`)
2. **Space Invaders** — ALE RAM→text; dense score
3. **Freeway** — ALE RAM→text; chicken-crossing score
4. **Demon Attack** — ALE RAM→text; dense score

New games onboard through the plugin contract — see `docs/PLUGIN_GUIDE.md`.

### Workshop tournament (honor system)

There is **no** shared multi-user eval / ELO / auto-ranking product. Teams
pick any game, screenshot theater or the scoreboard, and the instructor
ranks socially — scores are not comparable across games.

### Colab bake & stream

Instructors (or anyone with a GPU runtime) can bake packs with a live screen
stream in Google Colab — open [`colab_workshop.ipynb`](colab_workshop.ipynb)
(*File → Upload notebook* into Colab, or *Open in Colab* once the repo is on
GitHub). Pick a game, bake DQN + demos, watch frames update live, then push
to Hugging Face for the local playground **Teacher → dqn** field.

### Day-of install

Instructors: detect tier and print (or `--run`) the exact bring-up command:

```bash
python -m slm_rl.platform.launch
python -m slm_rl.platform.launch --run            # start playground
python -m slm_rl.platform.launch --run --docker   # compose instead
```

See `docs/LIFECYCLE.md` § Workshop day. Attendees open the browser only.

## Generation theater: watch the model before/after training

The workshop's money shot: play the base model and the current champion on
the SAME seeds, side by side ("stock vs trained, 0 → 1 A/B"), plus a
DQN-style grid of every generation. Eval episodes are never recorded on
disk, so `slm-rl theater` plays a small "exhibition" (10 seeded episodes per
side by default, one model in memory at a time) and writes both sides under
`<run_dir>/theater/{base,champion}/` in the exact `generations/gen_NNN/
rollouts/*.jsonl` layout the live-play viewer already understands — no
viewer code needed to "support" theater dirs.

```bash
uv run slm-rl theater --run-id <run-id>   # after `evolve` has promoted a champion
uv run slm-rl watch --run <run-id>        # or open the exhibition dirs directly
```

In the playground UI (`uv run slm-rl playground`), each scoreboard row has
an **A/B** button (launches the exhibition, then embeds base + champion
side by side with a live score strip) and a **gens** link
(`/gens/<name>/`, one viewer panel per generation the run has produced,
each filtered to that generation via `?gen=N`).

## Playground model picker

By default every experiment runs whatever model/backend your hardware tier
resolves to (`configs/hardware.yaml`). The playground's **New experiment**
card also has an optional **model** field (any Hugging Face repo id, e.g.
`Qwen/Qwen2.5-0.5B-Instruct`, or a local HF snapshot path)
and a **backend** select (`tier default`, `transformers`,
`transformers-4bit`, `mlx`) — leave both alone and
behavior is unchanged. Model-id validation is advisory only and never
blocks offline use: a quick local sanity check catches obvious typos
(whitespace, a bare word with no `/` that also isn't a real local path),
and a ~3s best-effort Hugging Face Hub lookup adds a non-blocking warning
("couldn't verify — offline?" or "not found on the Hub") to the create
response and the experiment's `experiment.json` — the experiment is
created either way. Every scoreboard row shows its resolved model (and
backend, if overridden) so A/B comparisons across model choices stay
legible. Guardrail: on the 8GB tier, stick to ≤1B-parameter models.

## Signup + publishing your work to Hugging Face

The first time the playground UI loads, it asks for a name and (optionally)
a Hugging Face token — stored **locally only**, in
`<home>/playground/profile.json` on the machine running the playground
(file permissions `0600`; never sent anywhere but the Hub, and only when
you click publish). This is not a multi-user server: every attendee runs
their own playground on their own laptop, so "the" profile is a single
local file, not an account system. Skipping the token just disables the
publish buttons with a tooltip — the playground stays fully usable
(rollouts/evolve/theater never need HF).

Once a token is on file, each scoreboard row gets a **publish** button that
pushes:
- the experiment's per-generation datasets to `{your-username}/slm-rl-<experiment>-data`
- the champion generation's LoRA adapter + a generated model card to
  `{your-username}/slm-rl-<experiment>` (skipped with a clear message if no
  generation has been promoted yet)

Both repos are created under your own account; the two sides are reported
independently, so a partial failure (e.g. dataset push works, model push
doesn't) is always visible rather than silently swallowed.

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

## Docker

Workshop path: start the stack, do **everything in the browser** (bake packs,
create projects, run games, evolve, theater, publish). No game CLI.

```bash
docker compose up --build
# UI:  http://127.0.0.1:5173/
# API: http://127.0.0.1:8780/
```

| In the UI | What it does |
|-----------|----------------|
| Projects → Bake workshop packs | Pre-bake demos / Atari DQN |
| New project | Pick name + game |
| Project → **Run game** | Quick teacher/solver episodes |
| Project → **Evolve** | Train (paste pack URL for day-of) |
| Project → Theater / Live view | Compare / watch |

Edit `web/` (Vite HMR) or `slm_rl/` / `configs/` (API process restarts via watchfiles) on the host — no image rebuild needed for those paths.

GPU host (CUDA + nvidia container toolkit):

```bash
docker compose --profile gpu up --build playground-gpu
# API on http://127.0.0.1:8781/ — start `web` too if you want the Vue UI:
# docker compose --profile gpu up --build
```

Run the test suite inside the container:

```bash
docker compose run --rm playground pytest
```

Notes:
- DQN teacher needs a **per-game** checkpoint: bake a pack (writes
  `runs/packs/<game>/dqn.pt`) or `slm-rl train-dqn --game <game> --out
  runs/teachers/dqn-<game>.pt`. The playground no longer points every game
  at Space Invaders.
- **Workshop packs:** bake from the Vue **Projects** page (no CLI). Packs
  land in `./runs/packs`; optional HF push uses the welcome-screen token.
  See `docs/LIFECYCLE.md` § Stage 2b.
- Experiment configs materialized *inside* the container hold container
  paths (`/app/runs/...`). If you switch between Docker and bare metal,
  re-create the experiment rather than reusing its config dir.
- The Vue app proxies `/api`, `/watch`, `/theater`, `/gens` to the
  `playground` service on the compose network.

### macOS (Apple Silicon)

The same `docker compose up --build` works on Docker Desktop — the image
builds natively for arm64 (llama.cpp is pinned to the portable
`armv8.2-a+dotprod+fp16` baseline, which every M-series chip supports).
The `gpu` profile is NVIDIA-only; skip it on a Mac.

Docker on macOS runs in a VM, so evolve inside the container is CPU-only —
fine for the workshop's quick-experiment loop. For Metal-accelerated evolve,
run natively instead:

```bash
uv sync --extra atari --extra mac --extra dev   # mlx-lm backend
uv run slm-rl playground
```

## Docs

- [`docs/LIFECYCLE.md`](docs/LIFECYCLE.md) — the end-to-end lifecycle: install → signup → experiment → evolve → compare → publish → end
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the full system design
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — design decisions D1–D10 with rationale
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — phased build plan
- [`docs/PLUGIN_GUIDE.md`](docs/PLUGIN_GUIDE.md) — how to add a game
