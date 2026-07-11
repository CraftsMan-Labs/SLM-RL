# Plan 005: Batch episode collection — K games per generate() call

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat cd2b4f4..HEAD -- slm_rl/rollout/ slm_rl/eval/suites.py slm_rl/inference/base.py slm_rl/agents/llm_agent.py`
> On any in-scope drift vs the "Current state" excerpts, STOP.

## Status

- **Priority**: P3
- **Effort**: L
- **Risk**: MED (touches the runner/monitor invariants every other test depends on)
- **Depends on**: none (but land after 001–004; it's a force multiplier, not a fix)
- **Category**: perf
- **Planned at**: commit `cd2b4f4`, 2026-07-11

## Why this matters

Rollout and eval collect episodes strictly serially: one `agent.act` → one
`backend.generate([messages], ...)` call per turn per episode. A 500-episode
eval is therefore ~2,000–4,000 sequential GPU generate calls, and evals
dominate wall-clock (measured: evals are >80% of a generation's runtime on
the 350M; worse on the 1.2B). Gymnasium's ecosystem answer is vectorized
environments — run K environments in lockstep and batch the policy forward
pass. The transformers backend already accepts a *list* of chats per
`generate` call, so batching K episodes cuts eval/rollout wall-clock by
roughly the effective batch factor (5–15× in practice) with zero effect on
what is measured — provided determinism and monitor semantics are preserved
per episode.

## Current state

- `slm_rl/inference/base.py` — `InferenceBackend.generate(self, chats, params)`
  takes a LIST of chat message-lists and returns a list of `GenOutput`
  (confirm by opening the file; `LLMAgent.act` calls
  `self.backend.generate([messages], self.params)[0]`).

- `slm_rl/rollout/runner.py` — `EpisodeRunner` drives ONE game: per-turn
  `agent.act(obs, history)` → `game.step` → `monitor.observe` →
  intervention handling (reflect nudge, mask_action menu filter, truncate) →
  `RolloutRecord` write. The retry ladder lives inside `LLMAgent.act`
  (a failed parse triggers a second generate call for that episode only).

- `slm_rl/eval/suites.py` — `run_suite(...)` loops seeds and builds one
  `EpisodeRunner` per seed, `pruner` pass-through, aggregates metrics.

- `slm_rl/orchestrator/generation.py` — rollout loop builds one
  `EpisodeRunner` per episode; eval goes through `run_suite`.

- Determinism contracts the batch must preserve: episode seeds fix the game;
  the pruner is deterministic per `(episode_seed, obs.turn)`;
  `LLMAgent._rng` fallback sampling is seeded per-agent. Records must be
  byte-equivalent in *content* (not order) to serial runs at temperature 0.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Tests | `uv run pytest -q` | all pass |
| Focused | `uv run pytest tests/test_batch_runner.py -q` | all pass (new file) |

## Scope

**In scope**:
- `slm_rl/rollout/batch_runner.py` (new)
- `slm_rl/eval/suites.py` (optional `batch_size` param routing to the batch runner)
- `slm_rl/orchestrator/generation.py` (use batch runner for rollout/eval when backend batches)
- `slm_rl/config/schema.py`, `configs/default.yaml` (`rollout_batch_size: int = 1`)
- `tests/test_batch_runner.py` (new)

**Out of scope** (do NOT touch):
- `slm_rl/rollout/runner.py` — the serial `EpisodeRunner` stays untouched as
  the reference implementation and the 8GB default path.
- `slm_rl/rollout/monitor.py` — one `DoomLoopMonitor` instance per episode;
  never shared.
- `slm_rl/agents/llm_agent.py` — do not refactor `act()`; the batch runner
  drives the backend directly (see Step 2).
- llama.cpp / MLX backends — batching is transformers/vLLM only; others keep
  `rollout_batch_size=1`.

## Git workflow

- Conventional commits + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer. No push unless instructed.

## Steps

### Step 1: Design constraint check (read-only)

Read `slm_rl/rollout/runner.py`, `slm_rl/agents/llm_agent.py`, and
`slm_rl/inference/base.py` fully. Confirm: (a) `generate` accepts multiple
chats; (b) all monitor/intervention/record logic is per-episode with no
cross-episode state; (c) the retry ladder is the only place a single turn
issues a second generate. If any of these is false → STOP.

**Verify**: write a 5-line summary of the three confirmations into the PR/commit description.

### Step 2: Implement `BatchedEpisodeRunner`

New file `slm_rl/rollout/batch_runner.py`. Shape:

- Constructor mirrors `EpisodeRunner` but takes `games: list[Game]`,
  `seeds: list[int]`, `episode_ids: list[str]`, shared `backend`,
  `system_prompt`, `gen_params`, plus the same `game_cfg/writer/pruner/...`.
- Maintains per-episode state tuples (game, monitor, obs, history, done,
  aggregates) — reuse the serial runner's per-step logic by extracting NO
  shared helpers (copy the ~40 lines; a `# ponytail:` comment noting the
  serial runner is the reference). Duplication is deliberate: the serial
  runner must stay untouched.
- Loop: gather `build_messages(system_prompt, obs_i)` for all live episodes →
  ONE `backend.generate(batch, params)` → per episode, run the parse ladder
  (`parse_action`), collecting retry-needed episodes into a second batched
  generate; after two failures use the seeded random fallback (one
  `random.Random(seed_i)` per episode, seeded at episode start).
- Step each live game, run its monitor, write its record, mark done episodes;
  exit when all done. Return the same per-episode summary dicts the serial
  runner returns, in input order.

**Verify**: `uv run pytest tests/test_batch_runner.py -q` (Step 4 tests) → pass.

### Step 3: Route rollout and eval through it

- `run_suite(..., batch_size: int = 1)`: when `>1`, chunk seeds and use the
  batch runner (fresh games per chunk), else the serial path. Metrics
  aggregation identical.
- `generation.py`: pass `cfg.train.rollout_batch_size` (new TrainConfig field,
  default 1) to rollout and `_eval`. The pruner alternation
  (`i % 10 < pruned_lot`) becomes per-episode within the chunk — compute the
  pruner per episode index exactly as today and pass a per-episode pruner
  list to the batch runner.
- `configs/default.yaml`: `rollout_batch_size: 1` (safe default), comment
  that CUDA tiers can set 16.

**Verify**: `uv run pytest tests/test_generation.py -q` → pass (defaults keep serial path).

### Step 4: Tests (equivalence is the whole game)

New `tests/test_batch_runner.py` with a `ScriptedBackend`-style fake that
serves batched requests (see `tests/test_parser.py` for the pattern):

1. **Serial-equivalence**: same 6 seeds through serial `EpisodeRunner` and
   `BatchedEpisodeRunner(batch=3)` with a deterministic fake backend →
   identical per-episode `outcome`, `steps`, `cum_reward`, and record
   `parsed_action` sequences.
2. **Ragged termination**: episodes ending at different turns; batch shrinks;
   all records written; no generate call contains a finished episode's prompt.
3. **Retry isolation**: fake backend that garbles episode 2's first output
   only → episode 2 gets `retry_ok`, episodes 1/3 unaffected, retry batch
   contained exactly one chat.
4. **Monitor isolation**: a degenerate script for one episode triggers its
   truncation; the other episodes' monitors record zero interventions.

**Verify**: `uv run pytest -q` → all pass.

## Test plan

Covered in Step 4 — the serial-equivalence test is the done-or-not-done test
for this plan.

## Done criteria

- [ ] `uv run pytest -q` exits 0 (including 4+ new batch tests)
- [ ] `git diff --stat` shows `slm_rl/rollout/runner.py` and `slm_rl/rollout/monitor.py` unchanged
- [ ] `grep -n "rollout_batch_size" slm_rl/config/schema.py configs/default.yaml slm_rl/orchestrator/generation.py` → ≥3 matches
- [ ] A 40-episode smoke (`uv run slm-rl eval --game mastermind --agent random --limit 40`) still passes serially
- [ ] `plans/README.md` status row updated

## STOP conditions

- Step 1 finds any cross-episode state in monitor/runner (invalidates the design).
- The serial-equivalence test cannot be made to pass without modifying
  `runner.py` (out of scope — report the mismatch instead).
- The transformers backend errors on >1 chat per generate (backend contract
  differs from `inference/base.py`'s signature promise).

## Maintenance notes

- The vLLM backend (Phase 2 roadmap) supersedes much of this for CUDA; keep
  the batch runner's interface backend-agnostic so vLLM slots in.
- Reviewer: scrutinize the retry second-batch logic — it's where episode
  cross-talk bugs would hide.
- Deferred: batching the *warm-start teacher* rollout (already engine-speed,
  no benefit) and llama.cpp batching (upstream server-mode question, not ours).
