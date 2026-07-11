# Plan 013: `slm-rl playground` — workshop coding UI (knobs + reward-code tab + experiment scoreboard)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving on. If
> any STOP condition occurs, stop and report — do not improvise. Commit in
> the worktree per the git workflow. Skip updating `plans/README.md` (the
> reviewer maintains the index). Read `CODING_GUIDELINE.md` at the repo root
> first — it is the contract this plan is reviewed against.
>
> **Drift check (run first)**: `git diff --stat de58ac6..HEAD -- slm_rl/ tests/ configs/` → must be empty. If not, STOP and report.
>
> **Environment note**: a GPU training run may be live on this machine. You
> must NEVER load a language model (no `--agent llm`, no transformers/vLLM
> model loads). All your subprocess tests use `--agent random` or
> `--agent solver` (CPU-only).

## Status

- **Priority**: P1 (user-requested: workshop UI to tweak reward functions and params)
- **Effort**: L
- **Risk**: MEDIUM (one small touch to the play path: an optional reward hook in the gym adapter — defaults preserve behavior exactly)
- **Depends on**: 007 (webui patterns), 008 (Space Invaders), 012 (DQN teacher) — all landed
- **Planned at**: commit `de58ac6`, 2026-07-11

## Why this matters

The user is running a workshop. Attendees should improve the system by
tweaking it: rebalance the reward, change selection pressure, swap the
heuristic teacher for the DQN one, loosen or tighten the anti-doom
thresholds — then *measure* whether their change helped, on a scoreboard,
against a baseline. The tight loop is: edit knobs (or reward code) → run a
quick CPU experiment (teacher/random rollouts, ~1–3 min) → compare mean
score / action mix / interventions across experiments. A "launch evolve"
button exists for configs that survive the quick screen (the real
rollout→train→eval loop, attendee's informed choice).

Both teachers are kept deliberately (user decision 2026-07-11): heuristic
vs DQN becomes a dropdown knob, and comparing them is a core workshop
exercise (they tie on score ~308 but differ hugely in behavior: NOOP 57%
vs 7%).

## Design decisions (already made — do not relitigate)

1. **Ship as `slm-rl playground`, stdlib-only server** (like `slm_rl/webui/`):
   `http.server` + `threading` + `json` only. Runs on attendee laptops under
   the 8GB floor with zero extra deps. Binds `127.0.0.1` by default.
2. **Experiments are materialized config directories.** Each experiment
   writes `runs/playground/<name>/config/{default.yaml, games/<game>.yaml}`
   (the repo configs deep-merged with the attendee's knob values) and then
   spawns a `rollout` subprocess pointed at that directory via a new
   `--config-dir` CLI option. Both loaders (`load_run_config`,
   `load_game_config` in `slm_rl/config/loader.py`) already accept
   `config_dir` — the CLI just doesn't expose it yet. This gives exact
   reproducibility for free: the experiment dir IS its config.
3. **Reward code tab = a reward hook file, not arbitrary core edits.** The
   gym adapter gains an optional `extra["reward_hook"]` (path to a Python
   file defining `shape_reward(ctx: dict) -> float`). Absent → byte-exact
   current behavior. The playground writes each experiment's edited hook to
   `runs/playground/<name>/reward_hook.py` and points the materialized game
   config at it. Executing user Python is by design — this is a local
   workshop tool on the attendee's own machine (same trust model as them
   editing the repo); document that in the module docstring.
4. **Gate purity is untouched.** The hook shapes the *training-time* reward
   (which drives reject_sft top-quantile selection — the pedagogical
   point). The eval gate's primary metric is mean **raw score** parsed from
   `outcome`, which the hook cannot alter. State this in the docs step.
5. **Comparability**: every experiment gets its own run-id
   (`pg-<name>`), so playground numbers are never compared against main
   runs. Reward-hook or knob changes inside the playground are exactly the
   "new run-id per boundary" rule, enforced by construction.
6. **Resource guards**: at most 1 quick-experiment subprocess and 1 evolve
   subprocess at a time (module-level locks); a second submission gets HTTP
   409 with a clear message. Workshop laptops are weak; queues invite
   confusion.

## Current state (read these files before coding)

- `slm_rl/config/loader.py` — `load_run_config(game, config_path, overrides,
  config_dir)` and `load_game_config(game, config_dir)`; both default
  `config_dir` to the repo `configs/`. `deep_merge` is here too. You will
  reuse `deep_merge` to materialize experiment configs.
- `slm_rl/cli.py` — `rollout` (line ~67) and `evolve` (line ~190) commands.
  `rollout` calls `load_run_config(game=game)` / `load_game_config(game)`
  with no config_dir; `evolve` calls `load_run_config(game=game,
  overrides=overrides)`. `cli.py` ends with `if __name__ == "__main__":
  app()` (line 370), so subprocesses can use
  `[sys.executable, "-m", "slm_rl.cli", ...]` — use that, not a PATH lookup
  of `slm-rl` (works in any venv layout).
- `slm_rl/bridges/gym_adapter.py` — `GymnasiumGameAdapter.__init__` reads
  `extra` keys with behavior-preserving defaults (lines 44–51); `step()`
  computes the decision reward at lines 128–135:

  ```python
  self._score += raw_sum
  reward = raw_sum / self.score_scale

  prev_lives = self._lives
  cur_lives = info.get("lives")
  if prev_lives is not None and cur_lives is not None and cur_lives < prev_lives:
      reward += self.life_loss_penalty  # penalty is negative, so add
  self._lives = cur_lives
  ```

  The hook wraps exactly this block's result (see Step 1). Monitor-side
  penalties (retry/fallback/truncate) are applied in the rollout runner,
  NOT here — they are out of the hook's reach by design; document that in
  the hook template.
- `slm_rl/webui/server.py` / `page.py` — the stdlib server pattern to
  imitate: `ThreadingHTTPServer`, a `_make_handler(...)` closure, a single
  `PAGE` string with inline JS polling JSON endpoints. Follow its style
  (docstring explaining read-only vs read-write surfaces, bounded
  resources, `log_message` silenced).
- Rollout JSONL layout: `runs/<run_id>/generations/gen_000/rollouts/*.jsonl`
  — **one file can hold ALL episodes** (records interleaved by write
  order). Group by `episode_id`; an episode's final record's
  `outcome` field is `"score:<int>"` for Atari. Per-record fields you need:
  `episode_id`, `parsed_action`, `monitor_flags` (dict, non-empty on
  intervention), `outcome` (only on the terminal record).
- `configs/games/space-invaders.yaml` — game knobs live here: `max_turns`,
  `monitor:` block (`action_repeat_threshold`, `ngram_loop_threshold`,
  `state_revisit_threshold`, `reward_stagnation_window`), `extra:` block
  (`env_id`, `action_repeat`, `score_scale`, `life_loss_penalty`,
  `noop_start_max`).
- `configs/default.yaml` — run knobs: `train.selection_quantile`,
  `train.episodes_per_generation`, `teacher.warmstart_episodes`,
  `teacher.dqn_checkpoint`.
- The trained DQN checkpoint exists at
  `runs/teachers/dqn-space-invaders.pt` on this machine (do not commit it;
  the knob default just points at that path).

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Env | `uv sync --extra cuda --extra dev --extra atari` | ok (once) |
| Tests | `uv run --no-sync pytest -q` | all pass (~45s baseline) |
| Focused | `uv run --no-sync pytest tests/test_playground.py -q` | all pass |
| Manual smoke | `uv run --no-sync slm-rl playground --game space-invaders --port 8899` | serves; Ctrl-C to stop |

## Scope

**In scope**:
- `slm_rl/playground/__init__.py`, `knobs.py`, `experiments.py`, `stats.py`,
  `server.py`, `page.py`, `reward_template.py` (all new)
- `slm_rl/bridges/gym_adapter.py` (reward hook only — ~15 lines)
- `slm_rl/cli.py` (`--config-dir` on `rollout` and `evolve`; new
  `playground` command)
- `tests/test_playground.py`, `tests/test_reward_hook.py` (new)
- `docs/ARCHITECTURE.md` (new short "Workshop playground" subsection)

**Out of scope** (do NOT touch):
- `slm_rl/training/`, `slm_rl/datagen/`, `slm_rl/eval/`,
  `slm_rl/orchestrator/` (except nothing — genuinely zero changes there),
  `slm_rl/teachers/`, `slm_rl/webui/` (the viewer stays a separate,
  read-only surface), `pyproject.toml`, `configs/*.yaml` (knob defaults are
  read from them, never written).

## Git workflow

Conventional commits + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
Never push. Never commit `note.txt`, `runs/`, `.venv`.

## Steps

### Step 1: Reward hook in the gym adapter

In `GymnasiumGameAdapter.__init__`, read `extra.get("reward_hook")` (path
string or None). If set: resolve at construction time — missing file →
`ValueError(f"reward_hook not found: {path!r}")` (never a silent fallback,
same doctrine as `dqn_checkpoint` in `make_teacher`). Load lazily on first
use via `importlib.util.spec_from_file_location`; cache the loaded
`shape_reward` callable on `self`. A module without a callable
`shape_reward` → `ValueError` on first step.

In `step()`, after the current reward computation (excerpt above), wrap:

```python
if self._reward_hook is not None:
    reward = float(self._shape(  # loaded shape_reward(ctx) -> float
        {
            "raw_points": raw_sum,          # ALE points this decision
            "default_reward": reward,        # what the built-in formula produced
            "score": self._score,            # cumulative raw score
            "lives_lost": lives_lost,        # bool: life lost this decision
            "lives": cur_lives,
            "turn": self._turn,
            "terminated": terminated,
            "truncated": truncated,
            "vector_obs": self.vector_obs(), # 128 floats, RAM/255
        }
    ))
```

(You will need to hold `lives_lost` in a local instead of the current
inline `if`.) `reward_hook` absent → the code path is identical to today —
add a test asserting byte-equal rewards over a full episode (Step 5).

### Step 2: `--config-dir` on `rollout` and `evolve`

Add `config_dir: str = typer.Option(None, help="Alternate configs/ root
(playground experiments)")` to both commands; thread as
`Path(config_dir)` into every `load_run_config(...)`/`load_game_config(...)`
call inside them (loader already accepts it; `None` → unchanged). Mirror
the existing override style — see `--selection-quantile` for the pattern.

### Step 3: Playground core (`knobs.py`, `experiments.py`, `stats.py`)

- `knobs.py` — a declarative `KNOBS: list[Knob]` (dataclass: `key`,
  `label`, `target` ∈ {"game", "game.monitor", "game.extra", "run.train",
  "run.teacher"}, `type`, `min`, `max`, `default_from` — read the actual
  default by loading the repo configs at request time, never hardcode).
  Ship at least: `max_turns`, `action_repeat`, `score_scale`,
  `life_loss_penalty`, `noop_start_max`, the four monitor thresholds,
  `selection_quantile`, `episodes_per_generation`, `warmstart_episodes`,
  and `teacher` (enum: `heuristic` | `dqn`; `dqn` materializes
  `teacher.dqn_checkpoint: runs/teachers/dqn-space-invaders.pt` into the
  experiment's default.yaml and passes `--dqn-checkpoint` to rollout).
  Plus experiment-level fields (not config): `agent` (`solver`|`random`),
  `episodes` (default 30, max 200), `seed` (default 20000).
- `experiments.py` — `create_experiment(home, name, knob_values,
  reward_code) -> ExperimentDir`: validate name (`[a-z0-9-]{1,40}`),
  deep_merge knob values into copies of the repo configs, write the
  materialized `config/` dir + `reward_hook.py` (if code given, after a
  `compile()` check — syntax errors are returned to the UI, not written),
  stamp `experiment.json` (knob values, created_at from the request, git
  SHA via `git rev-parse`). `launch_rollout(exp) -> subprocess.Popen`
  using `[sys.executable, "-m", "slm_rl.cli", "rollout", "--game", ...,
  "--config-dir", ..., "--run-id", f"pg-{name}", ...]`, stdout to
  `<exp>/rollout.log`. One module-level `threading.Lock` per subprocess
  kind (quick / evolve); busy → raise `Busy` (server maps to 409).
  `launch_evolve(exp, generations)` mirrors it with `--warm-start`.
- `stats.py` — `experiment_stats(run_dir) -> dict`: group records by
  `episode_id` (one JSONL holds many episodes — see Current state), return
  `{episodes, mean_score, median_score, max_score, action_mix (pct by
  parsed_action), intervention_episodes (count with any non-empty
  monitor_flags), status}`. Pure function over files; no caching.

### Step 4: Server + page (`server.py`, `page.py`, CLI command)

- Routes: `GET /` (PAGE); `GET /api/knobs` (schema + current defaults);
  `POST /api/experiments` (create + launch quick run; 409 when busy; 400
  on validation/syntax error with the message); `GET /api/experiments`
  (list with `experiment_stats` for each, including a synthetic `baseline`
  row = repo-default knobs, run lazily the first time the page loads —
  same lock); `GET /api/experiments/<name>/log` (last 50 lines of
  rollout/evolve log); `POST /api/experiments/<name>/evolve` (launch, 409
  when busy); `GET /api/reward-template` (the template text).
- `page.py` — single PAGE string, no external assets: left column = knob
  form (generated from `/api/knobs`) + a `<textarea>` code tab pre-filled
  from `/api/reward-template` (monospace, tab-key inserts spaces — 5 lines
  of JS); right column = scoreboard table (one row per experiment: name,
  episodes done, mean/median/max score, top-3 action mix, interventions,
  ▶ evolve button) polling `GET /api/experiments` every 3s. Keep the
  styling in the spirit of `webui/page.py` — dark, minimal, no framework.
- `reward_template.py` — `TEMPLATE: str`: a fully commented
  `shape_reward(ctx)` that reproduces the default formula from the ctx
  fields and shows two commented example tweaks (survival bonus per turn;
  harsher life-loss). State in comments: monitor penalties are applied
  outside this hook; the eval gate measures raw score and cannot be
  affected here.
- `cli.py`: `playground` command (`game` default `space-invaders`, `home`
  default `./runs`, `port` default `8780`, `host` default `127.0.0.1`),
  lazy-imports the playground server, prints the URL.

### Step 5: Tests

`tests/test_reward_hook.py` (`pytest.importorskip("ale_py")`):
1. **Hook-absent = byte-identical**: run 40 decisions with a fixed action
   script twice (no hook vs `extra` without the key) → identical reward
   sequences. Then monkey-check: config WITH a hook file returning
   `ctx["default_reward"]` → also identical (proves ctx carries the true
   default).
2. Hook returning `2 * ctx["default_reward"]` → every reward doubled.
3. Missing hook path → `ValueError` at construction; module without
   `shape_reward` → `ValueError` on first step.

`tests/test_playground.py` (no ALE needed except where marked):
4. `create_experiment` materializes a config dir whose
   `load_game_config(game, config_dir=...)` round-trips the knob values;
   name validation rejects `../evil`.
5. Reward code with a syntax error → rejected with a message, no file
   written.
6. `experiment_stats` on a hand-written synthetic JSONL (2 files, 3
   episodes interleaved, one with monitor_flags, scores 100/200/300) →
   mean 200, interventions 1, action_mix sums to ~100.
7. Busy lock: second `launch_rollout` while a fake Popen is alive →
   `Busy`.
8. HTTP: POST create → 200 and dir exists; second POST while busy → 409;
   GET /api/knobs returns every knob with a default. (Serve on port 0,
   real requests via `urllib`.)
9. (ALE) End-to-end: create experiment with `episodes=2, max_turns=40,
   agent=random`, launch, wait ≤120s, `experiment_stats` shows 2 episodes
   with integer scores.

**Verify**: `uv run --no-sync pytest -q` → all pass. Note the runtime of
`test_playground.py` — keep it under ~120s.

### Step 6: Manual smoke + docs

- `uv run --no-sync slm-rl playground --port 8899` → open page HTML via
  `curl -s localhost:8899/ | head -30`; `curl -s localhost:8899/api/knobs`
  → JSON with defaults matching `configs/games/space-invaders.yaml`.
  Create one real experiment via `curl -X POST` (teacher=dqn, 5 episodes,
  max_turns=200) and confirm the scoreboard JSON shows scores when done.
- `docs/ARCHITECTURE.md`: add a short "Workshop playground" subsection
  after the live-play viewer one: what it is, the materialized-config-dir
  reproducibility property, the reward hook seam + why gate purity is
  unaffected, the local-trust security model, and that it is a
  read-WRITE surface (unlike the viewer) but writes only under
  `runs/playground/`.

## Done criteria

- [ ] `uv run --no-sync pytest -q` exits 0
- [ ] `git diff --stat -- slm_rl/training/ slm_rl/datagen/ slm_rl/eval/ slm_rl/orchestrator/ slm_rl/teachers/ slm_rl/webui/ configs/ pyproject.toml` → empty
- [ ] `grep -rn "import gymnasium\|import ale_py\|import numpy" slm_rl/playground/` → empty (stdlib-only)
- [ ] Reward-hook-absent path proven byte-identical by test 1
- [ ] Manual smoke: knobs JSON + one real experiment scoreboard row
- [ ] All new-file docstrings state their read/write surface

## STOP conditions

- The reward computation in `gym_adapter.step()` has changed vs the excerpt
  in Current state (drift — report, do not adapt silently).
- You find yourself wanting to modify the rollout runner, monitor, or
  exporter to make stats work — the JSONL already carries everything;
  report instead.
- The HTTP tests are flaky under `ThreadingHTTPServer` on this machine —
  report with evidence rather than adding sleeps > 1s.

## Maintenance notes

- Knob defaults are read from the repo configs at request time, so future
  config changes flow into the UI automatically — nothing to maintain.
- When plan 011 (vision) lands, the playground's experiment runner needs no
  change (agent choice is a CLI flag); a "vision" agent option becomes one
  more enum value.
- The reward hook is Atari-adapter-only for now. If Mastermind should get
  one, that is a separate seam (its reward lives in the game env) — do not
  generalize preemptively.
