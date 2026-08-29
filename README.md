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

### Colab workshop notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CraftsMan-Labs/SLM-RL/blob/main/colab_workshop.ipynb)

[`colab_workshop.ipynb`](colab_workshop.ipynb) is a self-contained, chaptered
walkthrough of the whole platform for people who have never seen this repo. It
needs nothing but a free Colab **T4** runtime — no install, no Docker, no web
app. Every stage is driven by direct `slm_rl` calls in notebook cells, with the
playground's live viewers replaced by inline frames and plots:

games and text observations → config → a model making one decision → rollouts
and datasets → DQN teacher → packs → reject_sft and GRPO → eval and the
promotion gate → the full evolve loop → base-vs-champion theater → publishing to
Hugging Face → writing your own game.

Yellow Colab form cells drive the session: `MODE` (`QUICK` finishes each cell
in a minute or two; `FULL` is a real run), `PRECISION` (`q4` default, `fp16`,
or `auto`), plus the game, seed, and run name. Later chapters add bounded
knobs — temperature, gate margin, train strategy, theater seed — and short
predict-then-reveal quizzes. A few cells call `input()` on purpose so
`Runtime → Run all` pauses (join, parse guess, trainer, gate).
There is no participant-facing skip control, and blank answers do not continue.
Challenge cells are extra; skip those and the rest still runs.

#### Live progress (WorkShopTracker)

Participant progress is reported to
[workshop.craftsmanlabs.net](https://workshop.craftsmanlabs.net/signin) as
telemetry — not a grade.

1. Facilitator shares a **run join URL** (this cohort:
   `https://workshop.craftsmanlabs.net/join/slm-rl-test-run-056892`).
2. Attendee opens that URL, signs in, joins the run, then clicks **Generate key** on `/workshop`.
3. Attendee runs the **Load WorkShopTracker API key** cell and pastes the secret
   (`wsp_live_…`) into the `WST_API_KEY` form field (or stores it as Colab Secret
   `WST_API_KEY` / env). The value is never printed.
4. That cell connects via a small stdlib client in `lab.py`
   (no private-repo pip install), validates the key with `tracker.me()`, and
   later cells call `start_chapter` / `complete_chapter` for `chapter-0` …
   `chapter-13`.

**Never** put an admin/master key (`MASTER_KEY_NOTEBOOK` / `WST_ADMIN_KEY`) in
Colab. That secret is facilitator-only.

##### Facilitator: sync checkpoints + mint a run

Project id used for this workshop:
`97819f51-9100-43a0-a0de-330879218f16`.

Runs **snapshot** their checkpoint list when created. After syncing the 14
chapter keys below, create a **fresh** workshop run and put its join URL in the
notebook form / share it with the room.

```python
import os
import requests

BASE = "https://workshop.craftsmanlabs.net"
KEY = os.environ["WST_ADMIN_KEY"]  # facilitator master key — not for Colab
H = {"Authorization": f"Bearer {KEY}", "Accept": "application/json"}
PROJECT = "97819f51-9100-43a0-a0de-330879218f16"

chapters = [
    ("chapter-0", "Setup"),
    ("chapter-1", "The games"),
    ("chapter-2", "Config"),
    ("chapter-3", "The model plays"),
    ("chapter-4", "Rollout + dataset"),
    ("chapter-5", "Teachers / hybrid RL"),
    ("chapter-6", "Packs"),
    ("chapter-7", "Training"),
    ("chapter-8", "Eval and the gate"),
    ("chapter-9", "The evolve loop"),
    ("chapter-10", "Theater"),
    ("chapter-11", "Publish"),
    ("chapter-12", "Build your own game"),
    ("chapter-13", "Tests"),
]
for i, (key, title) in enumerate(chapters, start=1):
    requests.post(
        f"{BASE}/api/v1/projects/{PROJECT}/checkpoints",
        json={"key": key, "title": title, "sort_order": i},
        headers=H,
    ).raise_for_status()

run = requests.post(
    f"{BASE}/api/v1/workshop-runs",
    json={"project_id": PROJECT, "name": "SLM-RL cohort"},
    headers=H,
).json()
print("join URL:", run["join_url"])
# Open held chapters from the live dashboard, or:
# requests.patch(
#     f"{BASE}/api/v1/workshop-runs/{run['id']}/checkpoints/chapter-2",
#     json={"unlocked": True},
#     headers=H,
# ).raise_for_status()
```

If a checkpoint key already exists on the project, skip or update it in the
admin UI — duplicate keys return 409.

Run it side-by-side with the Vue deck. Each chapter heading names the matching
slides (`Presentation: …`). Pipeline diagrams are static SVGs, not Mermaid
source. Deck stills and short clips live in `docs/workshop/assets/deck` so Colab
can load them from this repo. Chapter 5 walks DQN on World 1-1 with recorded
improvement clips, then an optional live Mario demo; Chapter 6 explains how
teacher traces become curated SFT pairs. The Atari RAM-vector teacher remains
the critical path.

Note that a T4 is Turing and has no bf16 support, so the CUDA path selects fp16
by capability (`bf16_ok()` in `slm_rl/training/lora.py`); bf16 is used only on
Ampere and newer.

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
