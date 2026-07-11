# Plan 009: Heuristic Space Invaders teacher — rationale warm-start for the 350M

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat 3c84a51..HEAD -- slm_rl/teachers/ slm_rl/bridges/gym_adapter.py slm_rl/games/atari/ tests/test_si_teacher.py`
> On any in-scope drift vs the excerpts below, STOP.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED (heuristic quality determines warm-start value; measure, don't assume)
- **Depends on**: 008 (landed at `86b927c`)
- **Category**: direction (Space Invaders learning signal)
- **Planned at**: commit `3c84a51`, 2026-07-11

## Why this matters

Measured baseline (run `spaceinv-350m` gen 0, 2026-07-11): the raw 350M
spams one movement action until the doom-loop monitor truncates the episode
around decision 17 — the same disease Mastermind had, cured there by a
teacher warm-start with verbalized rationales (0.2%→1.2% on the 1.2B;
invalid 35%→0% on the 350M). Space Invaders has no exact solver, but it
doesn't need one: a ~30-line aim-and-fire heuristic over the decoded RAM
scores far above random, and — critically — its *reasoning* ("cannon at
x=72, block at x=88 → move right") is mechanical to verbalize, plan-002
style. This plan builds that teacher and wires it into the existing
warm-start path (`evolve --warm-start`), which is fully generic after plans
001/002: teacher rollout → verbatim-rationale SFT pairs → adopted
unconditionally as the RL init.

## Current state

- `slm_rl/teachers/__init__.py` — `make_teacher(game_cfg, seed)` /
  `make_pruner(game_cfg, top_k)`: both are
  `if game_cfg.name.startswith("mastermind"): ... raise ValueError`.
  `make_teacher` returns `(agent, "teacher:<name>")`; the `"teacher:"`
  model_id prefix is what makes `sft_export` keep completions verbatim
  (plan 002; `slm_rl/datagen/sft_export.py` line ~106).
- `slm_rl/teachers/mastermind_solver.py` — the exemplar teacher:
  `SolverAgent(Agent)` with `act(obs, history) -> ActionDecision(action,
  raw_completion=f"{rationale}\nACTION: {guess}",
  prompt_messages=build_messages(self.system_prompt, obs))`. Rationale
  contract: plain text, the string `ACTION:` appears exactly once, in the
  final line. Non-empty `prompt_messages` is REQUIRED or SFT export yields
  0 pairs.
- `slm_rl/orchestrator/generation.py` — teacher branch of `run_generation`
  is game-generic: `make_teacher(self.game_cfg, ...)`, no backend,
  `warmstart_episodes` episodes, forced `reject_sft`, adopted ungated
  (plan 001). The pruner is optional and mastermind-only — warm-start runs
  for this game must not pass `--pruner` (make_pruner raises; check how
  `GenerationRunner.__init__` builds `self.pruner`: it only does so when
  `cfg.teacher.pruner` is true, default false — so simply not passing
  `--pruner` suffices).
- `slm_rl/bridges/gym_adapter.py` (landed in 008) — `_observation()`
  builds `Observation(text, legal_actions, turn, metadata)` with
  `metadata = {"score": ..., "lives": ..., "decision": ...}`. The decoded
  RAM variables (player_x, enemies_x, invaders_left, missile_in_flight,
  missile_y) are computed inside `SpaceInvadersRenderer.render` from
  `ram_maps.space_invaders.decode(ram)` but are NOT in metadata — the
  teacher needs them there (Step 1). `decode()` lives in
  `slm_rl/games/atari/ram_maps/space_invaders.py`; `enemies_y` was
  empirically excluded (see its module docstring) — bomb positions are NOT
  available; the teacher cannot dodge specific bombs and must not pretend to.
- `slm_rl/agents/llm_agent.py` — module-level `build_messages(system_prompt,
  obs)` (used by teachers so their records carry LLM-identical prompts).
- `tests/test_solver.py` — the test pattern for teachers (win rate through
  the real `EpisodeRunner`, determinism per seed, records feed sft_export
  with pairs > 0, rationale contract assertions).

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Env | `uv sync --extra cuda --extra dev --extra atari` | ok (once) |
| Tests | `uv run --no-sync pytest -q` | all pass |
| Focused | `uv run --no-sync pytest tests/test_si_teacher.py -q` | all pass |
| Smoke | `uv run --no-sync slm-rl rollout --game space-invaders --agent solver --episodes 5` | mean episode reward clearly > random baseline; outcomes `score:<n>` |

## Scope

**In scope**:
- `slm_rl/teachers/space_invaders_heuristic.py` (new)
- `slm_rl/teachers/__init__.py` (add the space-invaders branch to `make_teacher`)
- `slm_rl/bridges/gym_adapter.py` (metadata gains the decoded dict — Step 1 only)
- `tests/test_si_teacher.py` (new)
- `docs/HYBRID_RL.md` (one sentence: first non-exact teacher landed)

**Out of scope** (do NOT touch):
- `slm_rl/teachers/pruner.py`, `make_pruner` — menu pruning has no
  Space Invaders analogue (6 actions is already a small menu).
- `slm_rl/orchestrator/`, `slm_rl/datagen/`, `slm_rl/training/` — the
  warm-start path is already generic; if it isn't, that's a STOP, not a fix.
- `slm_rl/games/atari/ram_maps/space_invaders.py` — no new offsets; use
  what 008 verified.
- `slm_rl/teachers/mastermind_solver.py`.

## Git workflow

- Conventional commits + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer. No push unless instructed.

## Steps

### Step 1: Expose the decoded state in Observation.metadata

In `gym_adapter.py`'s `_observation()`, merge the renderer's decoded
variables into metadata. Cleanest shape without changing the
`ObservationRenderer` ABC: call `decode` once in the adapter? No — decode is
game-specific and lives behind the renderer. Instead, extend the renderer
protocol *optionally*: if the renderer has a `decode(raw_obs) -> dict`
method (duck-typed, `getattr`), the adapter puts its result under
`metadata["state"]`. Give `SpaceInvadersRenderer` that method (it already
imports `ram_map.decode`; the render body should reuse it rather than
decoding twice). Mastermind is unaffected (different Game class entirely).

**Verify**: `uv run --no-sync pytest tests/test_space_invaders.py -q` →
pass; add a quick assertion to the existing metadata-related test (or a new
one) that `obs.metadata["state"]["player_x"]` is an int.

### Step 2: The heuristic teacher

`slm_rl/teachers/space_invaders_heuristic.py` — `HeuristicInvaderAgent(Agent)`:

- Ctor: `(system_prompt, seed=None)`; `self._rng = random.Random(seed)`
  (used ONLY for tie-breaks, so episodes stay deterministic per seed).
- `act(obs, history)`: read `s = obs.metadata["state"]`. Policy (aim-and-fire;
  keep it ~this simple, it does not need to be optimal — it needs to be
  *clearly better than random and legible*):
  - `dx = s["enemies_x"] + BLOCK_AIM_OFFSET - s["player_x"]` where
    `BLOCK_AIM_OFFSET` is a constant you calibrate empirically in Step 3
    (the block x is its left edge; aiming at the near column beats aiming
    at the edge).
  - If `abs(dx) <= AIM_TOLERANCE` (calibrate; start 4): action `FIRE` if
    missile ready else `NOOP`.
  - Else `RIGHTFIRE` if `dx > 0` else `LEFTFIRE` (move toward the block,
    firing opportunistically).
- Rationale (plan-002 contract: deterministic, one `ACTION:` occurrence,
  final line), from numbers in scope:
  - aligned: `f"My cannon is at x={px} and the invader block is at x={bx} — lined up. Missile {'in flight, waiting' if in_flight else 'ready: firing'}."`
  - moving: `f"My cannon is at x={px} but the invader block is at x={bx}, {abs(dx)} to the {'right' if dx>0 else 'left'} — moving that way and firing as I go."`
- Return `ActionDecision(action=<the ActionSpec from obs.legal_actions with
  that id>, raw_completion=f"{rationale}\nACTION: {action_id}",
  prompt_messages=build_messages(self.system_prompt, obs))`.
- Resolve ActionSpecs from `obs.legal_actions` by id — never construct your
  own (ids must match what the adapter maps back to ALE indices).

### Step 3: Calibrate + wire the factory

- Calibrate `BLOCK_AIM_OFFSET`/`AIM_TOLERANCE` with a scratch script (never
  committed): run the heuristic through `EpisodeRunner` on 20 seeds; try
  offsets in {0, 8, 16, 24} and tolerances in {2, 4, 8}; pick the best mean
  cum_reward. Record the small results table in your report. Expected: the
  teacher lands well above the random baseline (random ≈ 0.5–0.9 cum_reward
  per the 008 smoke; if the best heuristic config is not at least ~2× random,
  STOP and report — a weak teacher would poison the warm start).
- `make_teacher` in `teachers/__init__.py`: add
  `if game_cfg.name.startswith("space-invaders"):` → instantiate the game
  via `get_game(game_cfg.name)(game_cfg)` for its `system_prompt()`, return
  `(HeuristicInvaderAgent(prompt, seed), "teacher:space_invaders_heuristic")`.
  Keep lazy imports inside the branch (CODING_GUIDELINE 8GB rule — ale-py
  must not be imported unless this branch runs).

**Verify**: `uv run --no-sync slm-rl rollout --game space-invaders --agent solver --episodes 5` runs the heuristic (the `solver` agent name maps through `make_teacher` — confirm in `cli.py` and report if it doesn't) and prints clearly-above-random rewards.

### Step 4: Tests

`tests/test_si_teacher.py` (`pytest.importorskip("ale_py")` at top; follow
`tests/test_solver.py` patterns):

1. Teacher beats random: mean cum_reward over 10 seeded episodes through
   `EpisodeRunner` ≥ 2× a `RandomAgent` baseline over the same seeds.
2. Determinism: same seed → identical action sequence + `state_hash`.
3. Rationale contract: `completion.count("ACTION:") == 1`, last line is
   `ACTION: <valid id>`, `extract_action_token` recovers it, rationale
   contains at least one digit (the coordinates).
4. Records feed SFT: teacher episodes through `EpisodeRunner` + writer →
   `export_sft_dataset` yields pairs > 0 and at least one exported
   completion contains "invader block" (rationale survived verbatim via the
   `teacher:` prefix).
5. Monitor tolerance: over 10 seeds, zero doom-loop truncations (the
   heuristic legitimately repeats movement actions; the 008 thresholds must
   accommodate it — if not, report, don't retune the yaml silently).

**Verify**: `uv run --no-sync pytest -q` → all pass.

### Step 5: Doc line

`docs/HYBRID_RL.md`: one sentence under the teacher seam noting the first
non-exact (heuristic) teacher landed for Space Invaders, and that the DQN
teacher (`teachers/dqn.py`, ROADMAP Phase 3) remains the upgrade path.

**Verify**: `grep -n "space_invaders_heuristic\|heuristic teacher" docs/HYBRID_RL.md` → ≥1 match.

## Test plan

Covered in Step 4; test 1 (beats random) is the plan's reason to exist.

## Done criteria

- [ ] `uv run --no-sync pytest -q` exits 0
- [ ] Calibration table in the report; chosen constants as named module constants with the empirical values in a comment
- [ ] Smoke: 5 teacher episodes, mean reward ≥ 2× the random baseline from the same seeds
- [ ] `grep -n "space-invaders" slm_rl/teachers/__init__.py` → ≥1 match
- [ ] `git diff --stat -- slm_rl/orchestrator/ slm_rl/datagen/ slm_rl/training/ configs/` → empty
- [ ] `plans/README.md` status row updated

## STOP conditions

- The best calibrated heuristic is < 2× random (weak teacher — report;
  options like bomb-dodging need RAM offsets 008 couldn't verify, which is
  an operator decision).
- The warm-start path turns out NOT to be game-generic (something
  mastermind-specific in `run_generation`'s teacher branch).
- `rollout --agent solver` hardcodes the mastermind solver in `cli.py` in a
  way that can't route through `make_teacher` without touching out-of-scope
  files — report with the excerpt.
- Adding `metadata["state"]` breaks any existing test outside your new ones.

## Maintenance notes

- After this lands, the intended experiment is:
  `slm-rl evolve --game space-invaders --train-strategy reject_sft --warm-start --generations 3 --run-id spaceinv-350m-v2 --model LiquidAI/LFM2.5-350M --rollout-batch-size 16`
  (NO `--pruner` — mastermind-only). New run-id: prompt/data content changed.
- The heuristic ignores bombs (positions not decodable from verified RAM
  offsets) — expected life losses; its edge is aim quality. The DQN teacher
  over `vector_obs()` is the principled upgrade (ROADMAP Phase 3).
- If a future game adds a teacher, `make_teacher` is now two `if` branches —
  at three, refactor to a registry (note it, don't do it now).
