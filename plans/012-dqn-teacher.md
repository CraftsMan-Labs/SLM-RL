# Plan 012: CleanRL-pattern DQN teacher over RAM bytes (`teachers/dqn.py`)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving on. If
> any STOP condition occurs, stop and report — do not improvise. Commit in
> the worktree per the git workflow. Skip updating `plans/README.md` (the
> reviewer maintains the index). Audit every claim against actual tool output.
>
> **Drift check (run first)**: `git diff --stat <planned-at>..HEAD -- slm_rl/teachers/ slm_rl/cli.py slm_rl/config/schema.py configs/default.yaml tests/test_dqn_teacher.py` → only commits the reviewer lists as expected. `tests/test_dqn_teacher.py` must not exist.

## Status

- **Priority**: P1 (user-approved: "yes please and start building it")
- **Effort**: L
- **Risk**: MED (new training loop; quality of the artifact is verified AFTER background training, not in this task)
- **Depends on**: 008, 009 (landed)
- **Category**: direction (ROADMAP Phase 3: learned teachers; HYBRID_RL.md seams)
- **Planned at**: see reviewer note, 2026-07-11

## Why this matters

The hand-written heuristic teacher (plan 009) rescued Space Invaders warm-starts
but has measured flaws that hand-tuning won't close: left-wall camping when its
aim point is unreachable, 57% NOOP passivity while its missile flies. The
roadmap's answer is a learned classical teacher: a CleanRL-pattern DQN over
`Game.vector_obs()` (the 128 ALE RAM bytes, exposed for exactly this). It trains
at engine speed with no LLM in the loop, then serves the same three seams the
Mastermind solver validated: warm-start demonstrations (with Q-verbalized
rationales), top-k action menus, and V(s) shaping for a future Atari GRPO.
This task builds the machinery and proves its mechanics; the reviewer launches
the real (hours-long) training as a background run afterward and gates teacher
adoption on measured quality.

## Current state (verify each by reading the file)

- `slm_rl/teachers/__init__.py` — `make_teacher(game_cfg, seed=None)`:
  mastermind branch (SolverAgent) and space-invaders branch
  (`HeuristicInvaderAgent`), each lazy-importing inside the branch. Returns
  `(agent, "teacher:<name>")`.
- `slm_rl/teachers/space_invaders_heuristic.py` — the exemplar teacher agent:
  `act(obs, history) -> ActionDecision(action, raw_completion=rationale+"\nACTION: <id>",
  prompt_messages=build_messages(...))`; epsilon exploration via one
  `random.Random(seed)` advancing across episodes (plan 009's diversity
  lesson — a deterministic teacher on a near-seed-invariant env produces
  duplicate episodes that the SFT dedup quota collapses); honest exploration
  rationales; calibration constants with measured comments.
- `slm_rl/bridges/gym_adapter.py` — `GymnasiumGameAdapter`: `vector_obs()`
  returns the 128 RAM bytes / 255.0; `reset(seed)` (with seeded no-op starts
  when `extra.noop_start_max > 0`); `step(ActionSpec)` applies
  `action_repeat` env frames per decision, returns normalized reward
  (score/30 + life penalty). The DECISION level is the MDP the teacher acts
  in — train the DQN at the same granularity by driving the `Game` object,
  not a raw gym env.
- `slm_rl/games/atari/space_invaders.py` — `SpaceInvadersGame`,
  `game.reset(seed)`, `game.step(spec)`, `obs.legal_actions` (6 ActionSpecs,
  ids = ALE meanings), `game.vector_obs()`.
- `slm_rl/config/schema.py` — `TeacherConfig` (pruner..., warmstart_episodes...).
  New knob goes here + mirrored in `configs/default.yaml`.
- `slm_rl/orchestrator/generation.py` — teacher branch calls
  `make_teacher(self.game_cfg, ...)` (find the exact call + how cfg.teacher
  is available there).
- `slm_rl/cli.py` — typer app; `rollout --agent solver` routes through
  `make_teacher`; commands lazy-import heavy deps.
- torch is available via the `cuda` extra (installed); do NOT add
  dependencies. A GPU may be busy with a 350M run — default the DQN to CPU.

## Commands you will need

| Purpose | Command | Expected |
|---|---|---|
| Env | `uv sync --extra cuda --extra dev --extra atari` | ok (once) |
| Tests | `uv run --no-sync pytest -q` | all pass |
| Focused | `uv run --no-sync pytest tests/test_dqn_teacher.py -q` | all pass |
| Smoke train | `uv run --no-sync slm-rl train-dqn --game space-invaders --decisions 3000 --out /tmp/dqn-smoke.pt --device cpu` | checkpoint written, per-interval log lines |

## Scope

**In scope**:
- `slm_rl/teachers/dqn.py` (new — the whole feature)
- `slm_rl/teachers/__init__.py` (space-invaders branch prefers a DQN checkpoint)
- `slm_rl/config/schema.py` + `configs/default.yaml` (`TeacherConfig.dqn_checkpoint: str | None = None`, comment: path to a trained teacher checkpoint; None = heuristic/solver teachers)
- `slm_rl/orchestrator/generation.py` (ONLY: thread `cfg.teacher.dqn_checkpoint` into the `make_teacher` call)
- `slm_rl/cli.py` (new `train-dqn` command; thread `dqn_checkpoint` where `rollout --agent solver` builds the teacher — add a `--dqn-checkpoint` option there)
- `tests/test_dqn_teacher.py` (new)
- `docs/HYBRID_RL.md` (build-order: DQN teacher landed; usage note)

**Out of scope**: everything else — especially `slm_rl/training/` (the LLM
trainers), `slm_rl/datagen/`, the heuristic teacher's internals, and any
GRPO/V(s)-shaping wiring (that is a later plan; do not pre-build seams).

## Git workflow

Conventional commits + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer. No push.

## Steps

### Step 1: `slm_rl/teachers/dqn.py` — single-file CleanRL pattern

All torch imports lazy (module imports at function/class-init scope — the
module itself must import without torch). Three pieces:

1. **`QNet`** (lazy-defined or defined behind the import): MLP
   `obs_dim -> 256 -> 256 -> n_actions`, ReLU.
2. **`train_dqn(game_cfg, decisions, out_path, device="cpu", seed=0,
   log_every=500) -> dict`** — the classic loop, decision-granularity,
   driving the `Game` object (`get_game(game_cfg.name)(game_cfg)`):
   - Replay buffer (plain deque or preallocated arrays, capacity ~100k;
     store float32 obs to keep RAM ~100k×128×4×2 ≈ 100MB — 8GB-safe).
   - Epsilon-greedy: linear decay 1.0 → 0.05 over the first 50% of decisions.
   - γ=0.99, Adam lr 2.5e-4, batch 128, target-network sync every 1000
     updates, train every 4 decisions after a 1000-decision warmup, TD loss
     `smooth_l1(Q(s,a), r + γ·(1-done)·max_a' Q_target(s',a'))`.
   - Episodes reset with an incrementing seed (diversity via no-op starts).
   - Log every `log_every` decisions: `decisions=N episodes=E eps=X
     mean_ep_reward_last20=Y loss=Z` — ONE line each (these feed the
     reviewer's Monitor).
   - Checkpoint: `torch.save({"state_dict", "obs_dim", "n_actions",
     "action_ids" (the ActionSpec ids in Q-output order), "game",
     "decisions", "mean_ep_reward_last20"}, out_path)` — self-describing;
     also save every 25k decisions (crash tolerance, same path + ".tmp"
     then atomic rename).
   - Return the summary dict.
3. **`DQNTeacherAgent(Agent)`** — mirrors `HeuristicInvaderAgent`'s contract
   exactly:
   - Ctor `(checkpoint_path, system_prompt, seed=None)`: loads the
     checkpoint (CPU map_location), rebuilds QNet, `eval()` mode,
     `self._rng = random.Random(seed)`.
   - `act(obs, history)`: obs vector from... the agent receives
     `Observation`, not the game — get the vector from
     `obs.metadata["state"]`? NO: metadata carries decoded ints only. Add
     the RAM vector? Do NOT change the adapter for this — instead read
     `obs.metadata.get("vector_obs")`: check whether the adapter already
     exposes it; it does not. DECISION (made by the planner, implement
     as-is): extend `_observation()` in `gym_adapter.py` to include
     `metadata["vector_obs"] = self.vector_obs()` — a list of 128 floats
     per decision. This slightly grows records (prompt_messages dwarf it);
     acceptable, and it makes ALL RAM-based teachers/pruners possible
     without game-object access. Add `slm_rl/bridges/gym_adapter.py` to
     your in-scope list for this one addition.
   - Epsilon exploration at `EXPLORE_EPS = 0.05` (SAME diversity lesson as
     plan 009 — without it, a greedy DQN on a near-invariant env produces
     duplicate warm-start episodes): with prob eps, random legal action,
     honest rationale ("Trying {id} to vary my approach.").
   - Greedy branch: argmax over Q restricted to `obs.legal_actions` ids
     (map via the checkpoint's `action_ids`), rationale verbalizing the
     ranking honestly, e.g. `f"Q-values rank {best} highest ({q_best:.2f};
     next {second} {q_second:.2f})."` — single `ACTION:` occurrence, final
     line, digits present (same contract as 009's tests).
   - `model_id` string for records: `"teacher:space_invaders_dqn"`.

### Step 2: Factory + config threading

- `TeacherConfig.dqn_checkpoint: str | None = None` (+ yaml mirror, comment).
- `make_teacher(game_cfg, seed=None, dqn_checkpoint=None)`: in the
  space-invaders branch, if `dqn_checkpoint` is a readable file → return
  `(DQNTeacherAgent(...), "teacher:space_invaders_dqn")`; else the heuristic
  as today. A set-but-missing path is an ERROR (raise ValueError — silent
  fallback would poison a run the operator thought was DQN-taught).
- `generation.py`: pass `dqn_checkpoint=self.cfg.teacher.dqn_checkpoint`
  at the teacher-branch call site. `cli.py rollout`: add
  `--dqn-checkpoint` option threaded the same way.
- `cli.py`: new `train-dqn` command (`game`, `decisions` default 500_000,
  `out` required, `device` default cpu, `seed` default 0) — lazy import of
  `train_dqn`, prints the returned summary.

### Step 3: Tests (`tests/test_dqn_teacher.py`)

`pytest.importorskip("torch")`, `pytest.importorskip("ale_py")`. Keep the
whole file < ~90s runtime (tiny decision counts):

1. **Training mechanics**: `train_dqn(cfg, decisions=1500, out=tmp, device="cpu", seed=0)`
   → checkpoint exists; loss values finite (harvest from return dict);
   buffer filled; ≥2 episodes completed.
2. **Checkpoint round-trip**: load via `DQNTeacherAgent`; `act()` on a real
   reset obs returns a legal ActionSpec from the menu.
3. **Rationale contract** (mirror plan 009's): `count("ACTION:")==1`, last
   line, digits present, `extract_action_token` recovers the id; over 30
   decisions both greedy and exploratory rationales appear and exploratory
   ones never claim Q-reasoning.
4. **Determinism per construction**: two agents, same checkpoint + seed,
   same episode sequence → identical action sequences.
5. **Diversity**: one agent, 6 consecutive episodes → ≥5 distinct action
   sequences (the 009 regression, inherited).
6. **Factory**: `make_teacher(cfg, dqn_checkpoint=path)` → DQN teacher +
   correct model_id; `dqn_checkpoint=None` → heuristic (unchanged);
   nonexistent path → ValueError.
7. **vector_obs metadata**: after the adapter change, a reset Observation
   has `metadata["vector_obs"]` of length 128, floats in [0,1].

### Step 4: Smoke + docs

- CPU smoke: the 3,000-decision `train-dqn` command from the table —
  confirm log lines appear and the checkpoint loads. (No quality claim —
  3k decisions is mechanics-only.)
- `docs/HYBRID_RL.md`: mark the DQN-teacher build-order item landed; one
  usage paragraph: train via `slm-rl train-dqn`, adopt via
  `teacher.dqn_checkpoint`, quality gate belongs to the operator (compare
  vs heuristic mean before adopting).

**Verify**: `uv run --no-sync pytest -q` → all pass.

## Done criteria

- [ ] `uv run --no-sync pytest -q` exits 0 (new tests running, not skipped)
- [ ] `grep -n "dqn_checkpoint" slm_rl/config/schema.py configs/default.yaml slm_rl/teachers/__init__.py slm_rl/orchestrator/generation.py slm_rl/cli.py` → ≥5 hits
- [ ] Smoke train writes a loadable checkpoint; log lines match the documented format
- [ ] `python -c "import slm_rl.teachers.dqn"` works in an env WITHOUT torch? — cannot test here; instead: `grep -n "^import torch\|^from torch" slm_rl/teachers/dqn.py` → no top-level torch imports
- [ ] `git diff --stat -- slm_rl/training/ slm_rl/datagen/` → empty
- [ ] `plans/README.md` row updated (by reviewer)

## STOP conditions

- `vector_obs` cannot be exposed via metadata without breaking existing
  tests in a way you can't fix by updating ONLY new/related assertions.
- The Game-object step rate makes even 1,500 training decisions absurdly
  slow in tests (>60s) — report measured rate; don't silently shrink counts.
- You are tempted to wire V(s) shaping into GRPO or menus into the pruner —
  out of scope, later plan.

## Maintenance notes

- The operator (reviewer) will run the real training
  (`train-dqn --decisions 500000`, hours, background) and must compare the
  checkpoint's mean episode reward against the heuristic teacher (~7.9
  uncapped) before setting `dqn_checkpoint` on any run. Adoption is a
  measured decision, never automatic.
- Top-k Q menus (pruner seam) and V(s) GRPO shaping are the designed
  follow-ups — the self-describing checkpoint (action_ids, obs_dim) is
  what makes them drop-in later.
- If a second Atari game lands, `train_dqn` is already game-agnostic
  (drives the Game ABC); only the checkpoint path differs.
