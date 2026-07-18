# The full lifecycle: from a fresh machine to a published model

This is the end-to-end operator's journey — every stage a run goes through,
from installing the repo to publishing the trained result and retiring the
run. Companion docs: `PIPELINE.md` is the zoomed-in view of one generation
(stage 4 here), `ARCHITECTURE.md` is the layer map, `DECISIONS.md` holds
the D-numbered rationale each stage cites.

```
INSTALL ─▶ SIGNUP ─▶ EXPERIMENT ─▶ EVOLVE ──────────────▶ COMPARE ─▶ PUBLISH ─▶ END
(tier      (local     (knobs +      │ gen 0: baseline      (watch,    (HF: data   (resume,
 detect)    profile)   reward hook)  │ gen 1: warm-start?   theater    + adapter    export,
                                     │ gen 2+: RL loop      A/B, gens  + model      archive)
                                     ▼ promote / reject     grid)      card)
                                    runs/<run_id>/...
```

## Stage 0 — Install: the machine picks its own path

Nothing in the lifecycle is chosen by the user that the hardware can choose
instead. `configs/hardware.yaml` maps detected RAM/VRAM/MPS to a **tier**,
and the tier fixes three things at once: the default model, the inference
backend, and the training strategy.

| Tier | Model | Backend | Training |
|---|---|---|---|
| any-8gb (CPU laptop) | LFM2.5-350M | transformers | GRPO |
| mac-16gb | LFM2.5-1.2B-Instruct | transformers | GRPO |
| cuda-8-16gb | LFM2.5-1.2B-Instruct | transformers | GRPO |
| cuda-24gb | gemma-4-E2B-it | transformers | GRPO |

Two install routes:

- **Docker (workshop default):** `docker compose up --build` → playground
  on :8780. The image bundles `cpu-train`, so the whole loop works with no
  extra choices. `--profile gpu` for the CUDA image on :8781.
- **Bare metal:** `uv sync` with extras matched to intent —
  `--extra cpu-train --extra atari` (CPU transformers + GRPO/SFT),
  `--extra cuda --extra atari` (full GPU pipeline),
  `--extra mac` (Apple Silicon / optional MLX).

### Workshop day

On the morning of a workshop, instructors run the day-of launcher instead of
memorizing extras/profiles:

```bash
python -m slm_rl.platform.launch          # detect tier → print install + start
python -m slm_rl.platform.launch --run    # exec playground (never uv sync)
python -m slm_rl.platform.launch --run --docker   # exec compose bring-up
```

It uses `detect_host` / `resolve_tier` against `configs/hardware.yaml` and
prints the exact `uv sync --extra …` line and the matching
`docker compose` command (CPU image vs `--profile gpu`). Attendees stay
browser-only against that process — no in-UI driver installer.

Sanity check before anything else: `uv run slm-rl info` prints the detected
host, the resolved tier, and every registered game. If that line is wrong,
everything downstream is wrong — fix it first (`--tier` forces a tier).

Default tiers use `transformers` with PEFT adapters end-to-end. Optional
`--backend mlx` cannot yet hot-swap a freshly trained PEFT adapter (narrow
gap); stay on transformers for multi-generation evolve.

## Stage 1 — First launch and signup (playground)

`uv run slm-rl playground` (or the Docker port). On first load the UI asks
for a name and, optionally, a Hugging Face token. This is **local
onboarding, not an account system**: the profile lives in
`<home>/playground/profile.json`, written with `0600` permissions, token
never logged and only ever shown masked. Skipping the token disables the
publish buttons and nothing else. The token is validated lazily — at first
publish, not at signup — so signup works fully offline.

## Stage 2b — Workshop bake packs (instructor, days before)

Day-of attendees should **not** train DQNs or roll live teacher warm-starts.
Instructors bake packs in the **UI** after `docker compose up` (Projects page →
**Bake workshop packs**). Packs land under `./runs/packs/`. Optional public HF
push uses the profile token from Welcome.

**Day-of attendee path:** Welcome → HF credentials → create project → paste
**dataset URL** → evolve. Empty URLs → full DIY (live warm-start).

```bash
docker compose up --build   # UI :5173 — bake + train only here
```

## Stage 2 — Create an experiment

An experiment is a named, reproducible bundle of choices. The **New
experiment** card takes:

- **Knobs** — the training/monitor parameters worth touching, each with a
  tutorial card citing the measured numbers behind its default.
- **Reward code** — a browser-editable `shape_reward(ctx) -> float` hook.
  Each game documents its `ctx` keys (ALE score / lives / …). The hook
  shapes **training** rewards only;
  it is physically absent from the eval path (stage 4).
- **Model / backend (optional)** — any HF repo id or local path, plus a
  backend override. Left alone, the tier default applies. Validation is
  advisory: obvious typos block, "couldn't reach the Hub" only warns.
- **Teacher** — heuristic vs DQN where the game offers both.

Creating the experiment **materializes a full config dir** under
`runs/playground/<name>/` — a copy of `configs/` with the knob values,
hook path, and model choice baked in — and assigns run-id `pg-<name>`.
That materialized dir, not the repo's `configs/`, is what evolve reads
(`--config-dir`), so an experiment's numbers can never silently drift when
the repo defaults change. A changed prompt, reward, or eval protocol is a
**comparability boundary**: new experiment, new run-id, never a reused one.

**Workshop:** open the Vue project workspace and use **Run game** / **Evolve**
/ **Theater** — do not drive games from the terminal.

## Stage 3 — Gen 0: the baseline nobody skips

Before any training, `ensure_baseline()` evaluates the raw base model on
the game's **frozen eval suite** (seeded episodes, seeds ≥ 10000, disjoint
from every rollout seed) and caches the result as `generations/gen_000/`.
Gen 0 is usually terrible; that is the point. Every later promotion is
measured against a champion lineage that starts here, so "the model
improved" always has a denominator.

## Stage 4 — Evolve: the generation loop

Each generation runs `ROLLOUT → DATASET → TRAIN → EVAL → GATE` (the full
inner detail, including the mermaid diagram, is `PIPELINE.md`). The
lifecycle-level facts:

1. **(Optional) gen 1 is a teacher warm-start** (`--warm-start`): a
   scripted expert (Mastermind's exact solver, the ALE heuristics, or the
   DQN) plays `warmstart_episodes` games; the traces distill into the model
   via reject_sft and the result is **adopted unconditionally** as
   champion₁ — initialization, not a competitor (D12). Its eval is still
   run and recorded honestly.
2. **Gens 2+ are the RL loop.** The current champion plays
   `episodes_per_generation` episodes at temperature 0.8 (exploration);
   records land as JSONL. Seeds are derived per generation so no episode
   repeats and none collides with eval seeds. The doom-loop monitor rides
   along: reflect → backtrack → truncate-with-penalty, thresholds
   calibrated at ~2× measured competent-play maxima so good play is never
   flagged.
3. **Dataset:** rollouts consolidate to parquet; a **replay window** mixes
   in up to `replay_generations` previous generations' rollouts (teacher
   demos keep teaching after gen 1).
4. **Train:** the tier's strategy — `reject_sft` (keep the top
   `selection_quantile` of episodes by return, SFT on them) or `GRPO`
   (group-relative RL with `format_reward` + `deduction_reward`, entropy
   watchdog, KL anchor to the champion). Output: a PEFT LoRA adapter.
5. **Eval + gate:** the candidate adapter plays the same frozen suite at
   temperature 0.2, **with every training aid stripped** — no reward hook,
   no pruner, no teacher. `EvalGate.decide` promotes only if the primary
   metric beats the champion by `min_improvement` without regressing on
   invalid-action rate, intervention rate, or entropy. Promotion moves the
   `registry.json` champion pointer; rollback is the pointer simply not
   moving. When the pruner is enabled, a small `eval_pruned` side suite is
   recorded for curiosity — the gate never reads it.
6. **Remediation:** after `max_consecutive_failures` straight rejections,
   the learning rate halves (floored at 1e-6). A promotion resets it to
   the run-start value. (Entropy-bonus escalation was removed after one
   doubling produced entropy 7.82 — random play.)

On a single GPU the backend is created and closed **around each phase**, so
rollout, training, and eval never hold the GPU at once.

### What lands on disk

```
runs/<run_id>/
  run_config.yaml          frozen resolved config (written once)
  registry.json            champion pointer + promote/reject history + ELO
  generations/gen_NNN/
    rollouts/*.jsonl       every decision: state, rationale, action, reward
    dataset/train.parquet  consolidated training view (+ replay view)
    adapter/               PEFT LoRA adapter (unmerged)
    eval/results.json      frozen-suite metrics (champions only keep theirs)
    metrics.json           rollout/train/eval/gate summary for the gen
    MANIFEST.json          base model, parent gen, config hash, git sha
  theater/{base,champion}/ exhibition episodes (stage 5), viewer layout
```

Everything needed to answer "where did this model come from" is in the run
dir; nothing under `runs/` is ever committed to git.

## Stage 5 — Watch and compare

Observers are **read-only by construction** — they tail the record stream
and never feed back into training.

- **Live:** `uv run slm-rl watch --run <run-id>` (or the playground's
  embedded `/watch/<name>/` panel) streams episode cards with the model's
  rationales; ALE games get deterministic frame replay.
- **A/B theater** (`slm-rl theater --run-id <id>`, or the scoreboard's
  **A/B** button): plays the base model and the champion on the **same**
  exhibition seeds (20000+, disjoint from both eval and rollout), one model
  in memory at a time, and shows them side by side — the "stock vs trained,
  0 → 1" moment.
- **Generations grid** (`/gens/<name>/`): one viewer panel per generation,
  DQN-dashboard style — the whole lineage at once.
- **Numbers:** `slm-rl eval` re-runs the frozen suite ad hoc; `slm-rl elo`
  ranks generations for two-player games.

## Stage 6 — Publish

With a token on file, each scoreboard row's **publish** button (or
`slm-rl push-data` for datasets from the CLI) pushes to the *attendee's
own* HF account:

- per-generation datasets → `{username}/slm-rl-<experiment>-data`
- the champion's LoRA adapter + an auto-generated model card (base model,
  metrics, lineage from MANIFEST) → `{username}/slm-rl-<experiment>`

The two pushes report independently — a partial failure is visible, never
swallowed. If no generation has been promoted yet, the model push is
skipped with a clear message and the datasets still go. Evolve can also
stream datasets per generation as it runs (`--hf-repo`), best-effort,
never fatal to the loop.

## Stage 7 — Ending (and un-ending) a run

- **Stopping is free:** every generation is durable the moment its
  `metrics.json` is written. There is no "shutdown" step; kill the process.
- **Resuming is automatic:** `registry.json` knows the next unrun
  generation, so re-running `slm-rl evolve` with the same run-id continues
  where it left off (`--warm-start` self-skips past gen 1). One caution: if
  a generation was killed **mid-rollout**, archive its partial
  `rollouts/*.jsonl` first, or the re-run appends to it.
- **Exporting:** publish (stage 6) or the PEFT `adapter/` dir itself is the
  hand-off format; a future merge/export path may add other serving formats.
- **Retiring:** a run dir is self-contained — archive or delete it whole.
  Changing the game's prompts, rewards, physics, or eval protocol ends the
  run's comparability; start a new run-id rather than continuing.

## Invariants that hold across every stage

1. **Gate purity** — steering (reward hooks, pruners, teachers, monitor
   nudges) is never counted as model improvement; the gate eval runs bare.
2. **Seed discipline** — rollout, eval (≥ 10000), and exhibition (≥ 20000)
   seed ranges never overlap; eval seeds are frozen so every generation
   sits the same exam.
3. **Provenance** — every adapter carries a MANIFEST (config hash, git
   sha, parent generation); every run freezes its config at creation.
4. **Tokens stay local** — `profile.json` is 0600, masked in every HTTP
   response, sent only to the Hub on an explicit publish click.
5. **`runs/` is never in git** — weights live in the HF cache or the Hub,
   records live in the run dir, the repo holds only code and configs.

## Command crib sheet

| Stage | What to do |
|---|---|
| 0 | `docker compose up --build` |
| 1–2 | Browser `:5173` — Welcome → HF → Projects (bake packs in UI) |
| 3–4 | Project → **Run game** / **Evolve** (paste pack URL day-of) |
| 5 | **Theater** / **Live view** in UI |
| 6 | **Publish** in UI |
| 7 | Re-click Evolve (resume) · export (Phase 4, when shipped) |

Power-user / CI CLI (`slm-rl evolve`, `rollout`, …) still exists for tests —
**not** the workshop path.
