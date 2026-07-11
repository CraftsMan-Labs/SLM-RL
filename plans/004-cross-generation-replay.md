# Plan 004: Train each generation on a replay window of accumulated rollouts, not just its own

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat cd2b4f4..HEAD -- slm_rl/orchestrator/generation.py slm_rl/datagen/consolidate.py slm_rl/config/schema.py configs/default.yaml tests/test_generation.py`
> On any in-scope drift vs the "Current state" excerpts, STOP. (Note: plan 001
> legitimately edits the gate section of `generation.py` — that edit is
> expected drift; the rollout/dataset section this plan touches must match.)

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED (changes what every trainer sees; off-policy data enters GRPO — see step 3)
- **Depends on**: 001 (both edit `generation.py`; land 001 first)
- **Category**: direction (data efficiency)
- **Planned at**: commit `cd2b4f4`, 2026-07-11

## Why this matters

Every generation currently trains only on its own ~200 episodes; all prior
rollouts — including the 1,000 expert teacher episodes from the warm start —
are discarded after one use. Classical deep RL's core data-efficiency
mechanism (DQN's experience replay; cf. brendanator/atari-rl) is reusing
accumulated experience. For SLM-RL this specifically means: (a) the teacher's
expert demonstrations keep teaching for several generations instead of one,
and (b) rare format-mode wins — the exact behavior the gate measures — are
never thrown away. Expected effect: more wins per SFT batch, less
generation-to-generation variance.

## Current state

- `slm_rl/orchestrator/generation.py`, `run_generation`, DATASET section:

  ```python
  # 3. DATASET
  from slm_rl.datagen.consolidate import consolidate

  dataset = self.paths.dataset(generation)
  consolidate(self.paths.rollouts(generation), dataset)

  # 4. TRAIN
  strategy = create_strategy(strategy_name, self.cfg.train, self.model_id, self.game_cfg)
  result = strategy.train(dataset, self.paths.generation(generation), init_adapter=champ_adapter)
  ```

- `slm_rl/datagen/consolidate.py` — `consolidate(rollouts_dir, out_parquet, chunk_rows=...)`
  reads every `*.jsonl` under one directory into one parquet. Open the file
  and confirm the signature before editing.

- `slm_rl/orchestrator/paths.py` — `RunPaths.rollouts(gen)` / `.dataset(gen)` /
  `.generation(gen)` give per-generation dirs (`runs/<id>/generations/gen_NNN/...`).

- Both trainers consume the parquet through `_iter_records`
  (`slm_rl/datagen/sft_export.py:21`) which takes a single path.

- `TrainConfig` in `slm_rl/config/schema.py` is where knobs live, mirrored in
  `configs/default.yaml`.

- Tests: `tests/test_generation.py` builds runners with `make_runner(...)`
  (FakeBackend + FakeStrategy, `episodes_per_generation: 2`); FakeStrategy
  receives `dataset_path` — a natural probe point for what data the trainer
  was given.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Tests | `uv run pytest -q` | all pass |
| Focused | `uv run pytest tests/test_generation.py -q` | all pass |

## Scope

**In scope**:
- `slm_rl/orchestrator/generation.py` (DATASET section only)
- `slm_rl/config/schema.py`, `configs/default.yaml` (one field: `replay_generations`)
- `tests/test_generation.py`

**Out of scope**:
- `slm_rl/datagen/consolidate.py` — do not change its signature; build the
  replay view in the orchestrator by consolidating into a separate file.
- `slm_rl/datagen/sft_export.py`, `grpo_export.py` — they already handle
  multi-episode parquet; their caps/quotas do the recency balancing.
- The EVAL/GATE sections of `generation.py`.

## Git workflow

- Conventional commits + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer. No push unless instructed.

## Steps

### Step 1: Add the knob

`TrainConfig.replay_generations: int = 3` in `slm_rl/config/schema.py`
(docstring: "train on rollouts from the last N generations; 1 = current
behavior") and `replay_generations: 3` in `configs/default.yaml`.

**Verify**: `uv run pytest tests/test_generation.py -q` → pass (default only).

### Step 2: Build the replay dataset in the orchestrator

In the DATASET section of `run_generation`, keep the per-generation
consolidate (the per-gen parquet is a published artifact — HF push and the
dataset product depend on it), then build the training view:

```python
dataset = self.paths.dataset(generation)
consolidate(self.paths.rollouts(generation), dataset)

# replay window: this gen + up to N-1 previous gens that have rollouts
window = range(max(1, generation - self.cfg.train.replay_generations + 1), generation + 1)
sources = [self.paths.rollouts(g) for g in window if self.paths.rollouts(g).exists()]
train_view = dataset
if len(sources) > 1:
    train_view = self.paths.generation(generation) / "dataset" / "replay.parquet"
    with tempfile / staged dir approach: consolidate each source...
```

Implementation detail: `consolidate` takes one directory. The simplest
correct approach without touching its signature: create
`.../dataset/replay_src/` inside the current generation dir and **symlink**
each source rollout `*.jsonl` into it (`os.symlink`; names prefixed
`g{NNN}-` to avoid collisions), then `consolidate(replay_src, replay.parquet)`.
Pass `train_view` (not `dataset`) to `strategy.train`. Delete nothing.

Generation 0 doesn't exist as a rollout source (baseline is eval-only) —
the `exists()` filter handles it. Teacher rollouts (gen 1) enter the window
naturally.

**Verify**: `uv run pytest tests/test_generation.py -q` → pass.

### Step 3: Keep GRPO on-policy-ish — cap the window for GRPO

Off-policy prompts are fine for SFT but stale prompts weaken GRPO (the
policy that produced them differs from the one being trained; group samples
are generated fresh so rewards stay valid, but prompt distribution drifts).
`grpo_export` already prefers later turns under its 512-prompt cap; make it
prefer later *generations* too: in `slm_rl/datagen/grpo_export.py` the sort
key is `rows.sort(key=lambda t: t[0], reverse=True)` where `t[0]` is
`step_idx`. Records carry `generation`; change the collected tuple to
`(rec["generation"], rec["step_idx"])` and sort descending on both. This is
a 2-line change and is IN scope for this step only (exception to the
out-of-scope note above, limited to the sort key).

**Verify**: `uv run pytest tests/test_grpo_export.py -q` → pass (existing
tests use a single generation; add the new test in Step 4).

### Step 4: Tests

In `tests/test_generation.py`:
- `test_replay_window_feeds_trainer`: capture `dataset_path` passed to
  FakeStrategy (record it on the fake). Run gens 1 and 2 with
  `replay_generations=3`; assert gen 2's path ends with `replay.parquet` and
  that the parquet (or its source dir) contains records from both gen 1 and
  gen 2 (read with pyarrow; `generation` column has both values).
- `test_replay_disabled_is_current_behavior`: with `replay_generations=1`,
  gen 2's dataset path is the plain per-gen parquet.

In `tests/test_grpo_export.py`:
- `test_cap_prefers_recent_generations`: records from generation 1 and 2 with
  a monkeypatched `MAX_PROMPTS=2` → both kept rows come from generation 2.

**Verify**: `uv run pytest -q` → all pass.

## Test plan

Covered in Step 4; model new tests on the existing `make_runner` /
`write_jsonl` fixtures in the same files.

## Done criteria

- [ ] `uv run pytest -q` exits 0
- [ ] `grep -n "replay_generations" slm_rl/config/schema.py configs/default.yaml slm_rl/orchestrator/generation.py` → ≥3 matches
- [ ] Per-gen parquet still written for every generation (`dataset/train.parquet` path untouched — check the code)
- [ ] `plans/README.md` status row updated

## STOP conditions

- `consolidate`'s signature differs from `consolidate(rollouts_dir, out_parquet, ...)`.
- Symlinks are unavailable/fail on this filesystem (report; a copy fallback
  doubles disk for rollouts — a decision for the operator, not you).
- Plan 001 has not landed and you find merge conflicts in `generation.py`.

## Maintenance notes

- HF dataset push uploads `dataset/*` — `replay.parquet` will be uploaded
  too. That's acceptable (it's honest training provenance) but reviewers
  should know it appears.
- If generations grow large, the replay window is the memory-relevant knob
  (8GB rule: consolidation is chunked, but `select_episodes` loads records
  per episode — watch peak RSS in the CI memory gate when raising the window).
- Interaction with plan 003: replay × extra epochs multiplies train time;
  tune `replay_generations` first, epochs second.
