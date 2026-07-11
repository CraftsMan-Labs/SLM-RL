# Plan 008: Space Invaders via Gymnasium/ALE — first Atari game (RAM → text)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat 57c1f07..HEAD -- slm_rl/bridges/gym_adapter.py slm_rl/games/atari/ slm_rl/eval/suites.py configs/games/ tests/test_space_invaders.py`
> `slm_rl/games/atari/` must contain only `__init__.py`; `gym_adapter.py`
> must still be the Phase-3 stub. On any drift, STOP.

## Status

- **Priority**: P1 (user-requested)
- **Effort**: L
- **Risk**: MED (first external-env game; RAM offsets are empirical)
- **Depends on**: none (plans 001–007 landed at `57c1f07`)
- **Category**: direction (new game, Phase 3 opener)
- **Planned at**: commit `57c1f07`, 2026-07-11

## Why this matters

The user wants the loop to learn Space Invaders using Gymnasium
(https://github.com/Farama-Foundation/Gymnasium). The repo pre-designed this
path: `docs/PLUGIN_GUIDE.md` §6 (Atari recipe: RAM decoder → text renderer →
minimal action map + frame-skip), `slm_rl/bridges/gym_adapter.py` (Phase-3
stub with the intended `GymnasiumGameAdapter` + `ObservationRenderer` shape),
the `atari` extra in `pyproject.toml` (`ale-py`, `gymnasium` — already
declared, do not add deps), and `EvalSuite.primary_metric = "mean_score"`
(score games were anticipated; `run_suite` computes `mean_score` from
`cum_reward`). Space Invaders is dense-reward, which suits the eval gate:
mean score moves smoothly, unlike Mastermind's noisy 0–2% win band.
Training path for this plan is `reject_sft` (game-agnostic). GRPO is
mastermind-only (`grpo_export` raises for other games) — that stays true;
do not touch it.

## Current state

- `slm_rl/bridges/gym_adapter.py` (22 lines) — docstring promises: wraps a
  `gymnasium.Env` (ALE, `obs_type="ram"`) into the `Game` contract via a
  per-game `ObservationRenderer` (`render(raw_obs, info) -> (str,
  list[ActionSpec])`). `GymnasiumGameAdapter.__init__(config, opponent=None,
  env_id="", renderer=None)` raises NotImplementedError("Phase 3").
- `slm_rl/games/base.py` — `Game` ABC: `reset(seed) -> Observation`,
  `step(ActionSpec) -> StepResult`, `state_hash() -> str`,
  `system_prompt() -> str`, `eval_suite() -> EvalSuite` (classmethod,
  abstract), optional `vector_obs()`, `snapshot()/restore()` (default
  pickles `__dict__` — an ALE env handle is NOT picklable, see Step 2).
  `Observation(text, legal_actions, turn, metadata)`;
  `StepResult(observation, reward, terminated, truncated, info)`;
  `ActionSpec(id, label)`. Open the file and confirm the exact field
  names/signatures before coding.
- `slm_rl/games/mastermind/env.py` — the exemplar `Game` implementation
  (registration decorator, config `extra` reading, `eval_suite()`,
  outcome reporting via `info["outcome"]` on terminal steps).
- `slm_rl/eval/suites.py` — `EvalSuite(game, seeds, primary_metric,
  metadata)`; `primary_metric: str = "win_rate"  # or "mean_score"`;
  `run_suite` fills `metrics["mean_score"] = mean(cum_reward)` and
  `metrics["primary"] = metrics[suite.primary_metric]`.
- `slm_rl/rollout/runner.py` — the runner enforces nothing about turn caps
  itself; games self-terminate/truncate (verify by reading the loop). The
  runner records `outcome` from `result.info.get("outcome")` on terminal
  steps; `"score:<n>"` is a valid outcome (`slm_rl/datagen/schema.py`).
- Doom-loop monitor: `MonitorConfig.action_repeat_threshold` etc. — for
  Mastermind, repeating an action is always a mistake; for Atari it is
  legitimate play (holding LEFT, spamming FIRE). The game's yaml must relax
  these (Step 4).
- `pyproject.toml`: `atari = ["ale-py", "gymnasium"]` extra exists.
  ale-py ≥ 0.10 bundles the ROMs — no AutoROM step needed (verify in Step 0).
- Registration: `@register_game("space-invaders")` + an entry in
  `pyproject.toml` `[project.entry-points."slm_rl.games"]` is the pattern
  used by built-ins (PLUGIN_GUIDE §4) — BUT the atari extra may be absent,
  and `registry._scan_entry_points` already tolerates import failures
  ("must never break the platform"). Confirm that tolerance exists, then add
  the entry point.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Env setup | `uv sync --extra cuda --extra dev --extra atari` | installs gymnasium + ale-py |
| Tests | `uv run --no-sync pytest -q` | all pass (atari tests auto-skip if ale-py missing) |
| Focused | `uv run --no-sync pytest tests/test_space_invaders.py -q` | all pass |
| Smoke | `uv run --no-sync slm-rl rollout --game space-invaders --agent random --episodes 2` | 2 episodes, `outcome=score:<n>`, JSONL written |

## Scope

**In scope**:
- `slm_rl/bridges/gym_adapter.py` (implement the stub as designed)
- `slm_rl/games/atari/space_invaders.py`, `slm_rl/games/atari/ram_maps/space_invaders.py` (new; create `ram_maps/__init__.py`)
- `configs/games/space-invaders.yaml` (new)
- `pyproject.toml` (ONLY a `space-invaders` line in the existing `[project.entry-points."slm_rl.games"]` table — no dependency changes)
- `tests/test_space_invaders.py` (new)
- `docs/ROADMAP.md` (tick the Phase-3 line), `docs/PLUGIN_GUIDE.md` (one sentence noting Space Invaders landed as the first ALE game)

**Out of scope** (do NOT touch):
- `slm_rl/datagen/grpo_export.py`, `slm_rl/training/` — GRPO stays
  mastermind-only; Space Invaders trains via `reject_sft`.
- `slm_rl/teachers/` — the heuristic/DQN teacher is plan 009, not this one.
- `slm_rl/rollout/`, `slm_rl/eval/suites.py` logic (you only *use*
  `primary_metric="mean_score"`; do not modify suites.py),
  `slm_rl/orchestrator/`.
- `slm_rl/bridges/openenv_bridge.py`.

## Git workflow

- Conventional commits + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer. No push unless instructed.

## Steps

### Step 0: Environment + ROM verification (read-only)

`uv sync --extra cuda --extra dev --extra atari`, then verify ALE works and
determinism knobs exist:

```bash
uv run --no-sync python -c "
import gymnasium as gym, ale_py
gym.register_envs(ale_py)
e = gym.make('ALE/SpaceInvaders-v5', obs_type='ram', frameskip=4, repeat_action_probability=0.0)
ram, info = e.reset(seed=0)
print(ram.shape, e.action_space.n, info.get('lives'), e.unwrapped.get_action_meanings())
"
```

Expected: `(128,) 6 3 ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']`
(lives may be 3; order of meanings may differ — record what you see).
**`repeat_action_probability=0.0` is mandatory** — the v5 default of 0.25
(sticky actions) breaks the repo's seeded-determinism invariant
(CODING_GUIDELINE §1.4). If this arg is rejected or ROMs are missing, STOP.

### Step 1: Empirically verify the RAM map

The AtariARI benchmark (Anand et al. 2019) documents these Space Invaders
RAM offsets — treat them as HYPOTHESES to verify, not facts:
`player_x=28`, `enemies_x=26` (invader block x), `invaders_left_count=17`,
`missiles_y=9` (player missile), `num_lives=73`, `enemies_y=24`.

Write a throwaway probe in your scratch space (never committed): reset with
a fixed seed, hold RIGHT for ~20 decisions and assert `ram[28]` increases;
hold LEFT and assert it decreases; FIRE and watch `ram[17]` decrement when
reward arrives; compare `ram[73]` against `info["lives"]`. Record the probe
results in your report. Offsets that verify go into
`slm_rl/games/atari/ram_maps/space_invaders.py` as named constants with a
comment citing AtariARI + "verified empirically 2026-07". An offset that
does NOT verify: render without it (degrade the observation, don't guess) —
note it in the module docstring. `info["lives"]` is authoritative for lives
regardless; env `reward` is authoritative for score (never decode score
from RAM).

### Step 2: Implement `GymnasiumGameAdapter`

In `slm_rl/bridges/gym_adapter.py`, implement the class per its own
docstring, honoring the `Game` contract exactly:

- Ctor: build the env lazily on first `reset` (`import gymnasium`/`ale_py`
  inside the method — 8GB rule: core imports must not require the extra).
  Read from `config.extra`: `env_id`, `action_repeat` (default 3),
  `score_scale` (default 30.0), `life_loss_penalty` (default -0.5).
- `reset(seed)`: `env.reset(seed=seed)`; store RAM + info; turn = 0.
- `step(action)`: apply the chosen ALE action for `action_repeat`
  consecutive `env.step` calls (stop early on terminated/truncated),
  summing raw reward. `reward = raw_sum / score_scale`, minus
  `life_loss_penalty` when `info["lives"]` drops (add, it's negative).
  Track raw score in `self._score`. Truncate at `config.max_turns`
  decisions. On terminal/truncated steps set
  `info["outcome"] = f"score:{int(self._score)}"`.
- Observation: delegate to the renderer —
  `text, legal_actions = renderer.render(ram, info)`; metadata carries
  `{"score": ..., "lives": ..., "decision": turn}`.
- `state_hash()`: sha1 of RAM bytes + turn.
- `vector_obs()`: `[b / 255.0 for b in ram]` (enables the plan-009 DQN
  teacher).
- **Override `snapshot()/restore()`**: the base default pickles
  `__dict__`, and an ALE env handle is not picklable. Use
  `env.unwrapped.clone_state()` / `restore_state()` if straightforward;
  otherwise override both to `raise NotImplementedError("no backtrack for
  ALE games")` with a `# ponytail:` comment — the space-invaders yaml does
  not enable the backtrack intervention, so this is safe. Do NOT let the
  default pickle path stand.
- `system_prompt()` and `eval_suite()` stay abstract-ish here: implement
  them in the per-game subclass (adapter raises NotImplementedError with a
  pointer). Registry entries are per-game, not for the adapter.

### Step 3: The Space Invaders game + renderer

`slm_rl/games/atari/ram_maps/space_invaders.py`: verified offsets +
`decode(ram) -> dict` returning the named variables it could verify.

`slm_rl/games/atari/space_invaders.py`:

- `SpaceInvadersRenderer(ObservationRenderer)` — compact text (≤ 12 lines,
  8GB budget), e.g.:

  ```
  Score: 120. Lives: 3.
  Your cannon is at x=72 (0=far left, 185=far right).
  35 invaders remain; the block spans x=40-120, lowest row y=60.
  Your missile: in flight (y=30) | ready to fire.
  Move to dodge enemy bombs and line up shots.
  ```

  plus the 6 `ActionSpec`s with human labels
  (`ActionSpec(id="FIRE", label="fire")` etc. — ids are the ALE meaning
  strings; the adapter maps id → ALE action index from
  `get_action_meanings()` at reset).
- `@register_game("space-invaders") class SpaceInvadersGame(GymnasiumGameAdapter)`
  fixing `env_id="ALE/SpaceInvaders-v5"` and the renderer;
  `system_prompt()`: rules + strategy in ≤ ~110 words (shoot invaders for
  points; you lose a life when a bomb hits you — move away from bombs;
  repeating a movement action is fine, standing still under a bomb is not).
  Follow plan 002's convention: strategy sentences live in the system
  prompt.
- `eval_suite()`: `EvalSuite(game="space-invaders",
  seeds=tuple(range(10_000, 10_000 + 100)), primary_metric="mean_score")`.
- Entry point in `pyproject.toml`:
  `space-invaders = "slm_rl.games.atari.space_invaders:SpaceInvadersGame"`.
  First confirm `registry._scan_entry_points` tolerates a failing import
  (it must, for machines without the atari extra) — if it doesn't, STOP.

### Step 4: Config

`configs/games/space-invaders.yaml`:

```yaml
# First ALE game (plan 008). Score game: gate compares mean_score, not
# win_rate. Atari legitimately repeats actions (hold LEFT, spam FIRE), so
# the doom-loop thresholds are much looser than board games'.
max_turns: 80            # LLM decisions; x action_repeat(3) x frameskip(4) ~ 960 frames
eval_episodes: 100
eval_seeds_start: 10000
monitor:
  interventions: [reflect, truncate]
  action_repeat_threshold: 12
  ngram_loop_threshold: 8
  state_revisit_threshold: 6
extra:
  env_id: ALE/SpaceInvaders-v5
  action_repeat: 3
  score_scale: 30.0        # max single-invader value; keeps per-step reward <= ~1
  life_loss_penalty: -0.5
```

Adjust monitor keys to the real `MonitorConfig` fields (open
`slm_rl/config/schema.py`).

### Step 5: Tests

`tests/test_space_invaders.py`, top:
`pytest.importorskip("ale_py"); pytest.importorskip("gymnasium")` so the
suite stays green on machines without the extra. Cases:

1. Registration + config load: `get_game("space-invaders")` resolves;
   `load_game_config("space-invaders").extra["env_id"]` correct.
2. Determinism: two games, same seed, identical `state_hash()` after reset
   and after the same 5-action script; different seed → different hash.
3. Menu: 6 legal actions, ids match the env's action meanings, adapter
   resolves each id to a step without error.
4. Reward + outcome: run a scripted episode to `max_turns` with NOOP —
   truncated, `info["outcome"]` starts with `"score:"`; reward for a
   no-score step is 0.0 (minus nothing).
5. Renderer budget: observation text ≤ 12 lines and < 600 characters;
   mentions lives and cannon position.
6. `vector_obs()` length 128, all values in [0, 1].
7. EpisodeRunner integration: `RandomAgent` through the real
   `EpisodeRunner` for 2 seeded episodes — records written, no crash,
   monitor does not truncate a NOOP-free random script before turn 20
   (this is the loosened-thresholds regression test).

### Step 6: Smoke + docs

`uv run --no-sync slm-rl rollout --game space-invaders --agent random --episodes 2`
→ both episodes end with `score:<n>` outcomes (any n ≥ 0), JSONL valid.
Tick the ROADMAP Phase-3 line; add one sentence to PLUGIN_GUIDE §6 noting
Space Invaders landed via this recipe (out-of-ramp-order by user request).

**Verify**: `uv run --no-sync pytest -q` → all pass.

## Test plan

Covered in Step 5; the determinism and monitor-threshold tests are the
regression anchors.

## Done criteria

- [ ] `uv run --no-sync pytest -q` exits 0 (atari tests running, not skipped, in your worktree)
- [ ] Step 0 one-liner prints the 128-byte RAM shape and 6 actions
- [ ] Smoke: 2 random-agent episodes end `outcome=score:<n>`
- [ ] `grep -n "space-invaders" pyproject.toml` → 1 match (entry point only; dependency sections untouched)
- [ ] `git diff --stat -- slm_rl/datagen/ slm_rl/training/ slm_rl/teachers/ slm_rl/rollout/ slm_rl/orchestrator/ slm_rl/eval/` → empty
- [ ] RAM-probe results (which offsets verified) recorded in your report
- [ ] `plans/README.md` status row updated

## STOP conditions

- ROMs unavailable or `repeat_action_probability=0.0` rejected (Step 0).
- `Game`/`Observation`/`StepResult` field names differ materially from the
  Current state description (drift — reread base.py, report).
- `registry._scan_entry_points` does NOT tolerate failing imports (adding
  the entry point would break atari-less installs).
- None of the AtariARI offsets verify empirically (the renderer would be
  score+lives only — that's below the bar for "teach it to play"; report).
- You are tempted to modify `run_suite`, the monitor, or GRPO export.

## Maintenance notes

- Space Invaders numbers are mean_score, not win_rate — never compare to
  Mastermind metrics; the gate's `min_improvement=0.01` margin is in
  *normalized score* units here (~1/3 of one invader kill per episode);
  operators may need a larger margin once baselines are measured.
- Plan 009 (next): heuristic teacher (aim-at-block + dodge-bombs) emitting
  plan-002-style rationales for `reject_sft` warm-start; later the
  ROADMAP's `teachers/dqn.py` over `vector_obs()`.
- Wall-clock: 100-episode eval × 80 decisions is ~8k generate calls —
  set `rollout_batch_size: 16` (plan 005) on CUDA tiers for Atari runs.
- `pruner`/`eval_pruned` are Mastermind-only concepts; `make_pruner` raises
  for this game — evolve runs must not pass `--pruner`.
