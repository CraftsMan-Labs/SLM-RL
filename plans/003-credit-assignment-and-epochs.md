# Plan 003: Return-weighted SFT selection + a second GRPO epoch (and document why terminal-reward grafting onto GRPO is a no-op)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat cd2b4f4..HEAD -- slm_rl/datagen/sft_export.py slm_rl/training/grpo.py slm_rl/config/schema.py configs/default.yaml tests/test_sft_export.py docs/HYBRID_RL.md`
> On any in-scope drift vs the "Current state" excerpts, STOP.

## Status

- **Priority**: P2
- **Effort**: S
- **Risk**: LOW
- **Depends on**: 002 (both edit `sft_export.py`; land 002 first)
- **Category**: direction (training-signal quality)
- **Planned at**: commit `cd2b4f4`, 2026-07-11

## Why this matters

Classic policy-gradient practice (Karpathy's pong-from-pixels) assigns every
action a *discounted share of the episode outcome* and takes many gradient
updates over a large batch. SLM-RL's audit found two small, real gaps and one
tempting-but-wrong idea:

1. **SFT selection ignores time-to-win.** A 4-turn win and an 11-turn win
   contribute pairs with equal weight; short wins are strictly better
   demonstrations.
2. **GRPO takes one pass over ≤512 prompts per generation** — very few
   updates per unit of experience (Karpathy: ~800 updates over 200k games).
3. **Tempting but wrong**: stamping a discounted terminal reward into
   `game_ctx` and adding it as a GRPO reward function does nothing — the
   terminal outcome comes from the *historical* episode, so it is constant
   across the group's 8 sampled completions for the same prompt, and GRPO's
   group-normalized advantage `(r − mean(group)) / std(group)` cancels any
   per-prompt constant exactly. Cross-turn credit in this codebase is
   correctly carried by the potential-shaped elimination reward
   (Φ(s) = −log|consistent(s)|), which telescopes to the terminal goal.
   This plan documents that so nobody re-attempts it.

## Current state

- `slm_rl/datagen/sft_export.py` — `select_episodes(dataset_path, cfg)`
  (lines 42–86) selects ALL winning episodes plus the top
  `cfg.selection_quantile` by return, applies a monitor-clean filter and a
  diversity quota (`cfg.max_duplicate_action_sequences`). Episodes are lists
  of step dicts with `outcome` on the terminal record and `step_idx` ordering.
  Nothing in the selection or export weights short wins above long ones.

- `slm_rl/training/grpo.py` — `GRPOConfig(... num_train_epochs=1, ...)`
  (in `GRPOStrategy.train`, ~line 160):

  ```python
  learning_rate=self.cfg.learning_rate,
  num_train_epochs=1,
  per_device_train_batch_size=8 if cuda else 2,
  ```

- `slm_rl/config/schema.py` — `TrainConfig` holds the training knobs
  (`group_size`, `entropy_floor`, `episodes_per_generation`, ...). New knobs
  go here + mirrored in `configs/default.yaml`.

- `docs/HYBRID_RL.md` — "Four integration seams" section; seam 3 describes
  potential-based shaping.

- Convention: deliberate simplifications carry `# ponytail:` comments.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Tests | `uv run pytest -q` | all pass |
| Focused | `uv run pytest tests/test_sft_export.py -q` | all pass |

## Scope

**In scope**:
- `slm_rl/datagen/sft_export.py`
- `slm_rl/training/grpo.py` (one line: epochs)
- `slm_rl/config/schema.py`, `configs/default.yaml` (two new TrainConfig fields)
- `tests/test_sft_export.py`
- `docs/HYBRID_RL.md` (one paragraph)

**Out of scope**:
- `slm_rl/datagen/grpo_export.py` and the GRPO reward functions — per the
  no-op analysis above, do NOT add terminal rewards there.
- `slm_rl/training/reject_sft.py` — it consumes the exported JSONL; no change.

## Git workflow

- Conventional commits + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer. No push unless instructed.

## Steps

### Step 1: Prefer short wins in SFT selection

Add `TrainConfig.win_turn_cap: int = 0` (0 = disabled) to
`slm_rl/config/schema.py` and `configs/default.yaml` (`win_turn_cap: 0`).
In `select_episodes`, after the win/top-quantile selection, when
`cfg.win_turn_cap > 0` drop winning episodes longer than `win_turn_cap`
steps, **unless that would leave zero winning episodes** (then keep the
shortest one — add a `# ponytail:` comment). Rationale: a 10-turn win is a
worse demonstration than a 4-turn win; the solver averages ~4–5 turns.

**Verify**: `uv run pytest tests/test_sft_export.py -q` → pass.

### Step 2: Weight later-turn pairs of wins upward (discounted return, ponytail form)

Full per-sample loss weights aren't supported by the SFT path; approximate
discounted return-to-go by *duplication*: in `export_sft_dataset`, when the
episode's outcome is `"win"` and `cfg.sft_win_final_dup > 1` (new
`TrainConfig` field, default `1` = disabled), write the **final decision pair
of each winning episode** `sft_win_final_dup` times. The final pair is the
one whose next feedback is the win — the highest-signal demonstration
(discount γ^0). Mark with `# ponytail: duplication ≈ sample weighting; real
per-sample weights if the trainer ever supports them`.

**Verify**: new test in Step 4 passes.

### Step 3: Two GRPO epochs

In `slm_rl/training/grpo.py`, change `num_train_epochs=1` to
`num_train_epochs=2` with a one-line comment: more updates per unit of
experience; entropy watchdog + KL-to-champion bound the extra drift.

**Verify**: `uv run pytest tests/test_grpo_rewards.py -q` → pass (rewards
untouched; this is config only).

### Step 4: Tests + doc paragraph

- `tests/test_sft_export.py`: (a) with `win_turn_cap=5`, an 8-step winning
  episode is excluded while a 4-step one survives; with only long wins, the
  shortest is kept. (b) with `sft_win_final_dup=3`, the final pair of a
  winning episode appears exactly 3 times and other pairs once.
- `docs/HYBRID_RL.md`: under seam 3, add a short paragraph titled
  "Why we don't graft terminal rewards onto GRPO": the group-advantage
  constant-cancellation argument from "Why this matters" (2–4 sentences).

**Verify**: `uv run pytest -q` → all pass; `grep -n "terminal rewards onto GRPO" docs/HYBRID_RL.md` → 1 match.

## Test plan

Step 4(a) and 4(b) are the regression tests; model them on existing fixtures
in `tests/test_sft_export.py` (synthetic record dicts written to JSONL).

## Done criteria

- [ ] `uv run pytest -q` exits 0
- [ ] `grep -n "win_turn_cap\|sft_win_final_dup" slm_rl/config/schema.py configs/default.yaml` → 4 matches (2 per file)
- [ ] `grep -n "num_train_epochs=2" slm_rl/training/grpo.py` → 1 match
- [ ] `grep -rn "terminal" slm_rl/datagen/grpo_export.py slm_rl/training/grpo.py | grep -iv "comment\|#"` adds no new reward code (no terminal reward func was added)
- [ ] `plans/README.md` status row updated

## STOP conditions

- `select_episodes` signature or selection logic differs materially from the
  description (drift).
- Adding the second GRPO epoch makes `test_grpo_*` fail (unexpected coupling).
- You are tempted to implement per-sample loss weights inside
  `reject_sft.py` — that's out of scope; report instead.

## Maintenance notes

- If TRL's SFT path later supports per-sample weights, replace Step 2's
  duplication with real weights (the ponytail comment marks the site).
- Reviewer: check that `win_turn_cap`/`sft_win_final_dup` default to disabled
  so existing runs reproduce identically until opted in.
- The epochs change interacts with plan 004 (replay): more data × more epochs
  raises train time per generation — watch wall-clock in the first combined run.
