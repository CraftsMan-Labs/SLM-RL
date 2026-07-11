# Plan 006: Register mastermind-easy (64 codes) as the curriculum entry point

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat cd2b4f4..HEAD -- slm_rl/games/mastermind/env.py slm_rl/datagen/grpo_export.py slm_rl/teachers/__init__.py configs/games/ tests/test_curriculum.py`
> On any in-scope drift vs the "Current state" excerpts, STOP.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: direction (curriculum / measurement resolution)
- **Planned at**: commit `cd2b4f4`, 2026-07-11

## Why this matters

On standard Mastermind (4 pegs × 6 colors = 1,296 codes) the project's models
live in the 0–2% win band, where the 500-episode eval suite's noise
(σ ≈ 0.45% at p=1%) swamps real effects — two identical warm-start runs
measured 1.6% and 0.4% (July 2026). A 3-peg × 4-color variant has 64 codes:
legal-random play wins ~19%, deduction wins ~100%, so effect sizes grow
~20× and every experiment becomes statistically readable. This is the
standard RL curriculum move (start where the gradient is dense, ramp
difficulty). The game engine already parameterizes size via `config.extra`;
what's missing is a registered name, a config file, and the relaxation of two
hardcoded `== "mastermind"` checks.

## Current state

- `slm_rl/games/mastermind/env.py` — `MastermindGame.__init__` reads
  `extra.get("code_length", 4)`, `extra.get("num_colors", 6)`,
  `extra.get("allow_duplicates", True)`; registration is
  `@register_game("mastermind")` on the class. `eval_suite()` returns
  `EvalSuite(game="mastermind", seeds=tuple(range(10_000, 10_500)), ...)`.

- `slm_rl/games/registry.py` — `register_game(name)` is a decorator factory;
  `get_game(name)` resolves. (Open it: registering the same class under a
  second name via a direct call is expected to work —
  `register_game("x")(Cls)`.)

- `slm_rl/config/loader.py` — `load_game_config(game)` reads
  `configs/games/<game>.yaml` and merges `{"name": game}`.

- Two hardcoded name checks block a variant:

  ```python
  # slm_rl/datagen/grpo_export.py (~line 26)
  if game_cfg.name != "mastermind":
      raise NotImplementedError(...)

  # slm_rl/teachers/__init__.py (both factories)
  if game_cfg.name == "mastermind":
  ```

- `configs/games/mastermind.yaml` is the exemplar config:

  ```yaml
  max_turns: 12
  eval_episodes: 500
  eval_seeds_start: 10000
  monitor:
    interventions: [reflect, truncate]
    action_repeat_threshold: 2
  extra:
    code_length: 4
    num_colors: 6
    allow_duplicates: true
  ```

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Tests | `uv run pytest -q` | all pass |
| Smoke | `uv run slm-rl rollout --game mastermind-easy --agent solver --episodes 10` | win rate 10/10, valid JSONL path printed |

## Scope

**In scope**:
- `slm_rl/games/mastermind/env.py` (alias registration + eval_suite game name)
- `configs/games/mastermind-easy.yaml` (new)
- `slm_rl/datagen/grpo_export.py`, `slm_rl/teachers/__init__.py` (name checks)
- `tests/test_curriculum.py` (new)

**Out of scope**:
- `configs/games/mastermind.yaml` — the standard game is unchanged.
- `pyproject.toml` entry points — in-repo registry registration suffices;
  only add an entry point if `slm-rl info` is required to list the variant
  and doesn't (check first).
- Any automatic difficulty-ramp logic — promotion-triggered ramping is
  explicitly deferred (see Maintenance notes).

## Git workflow

- Conventional commits + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer. No push unless instructed.

## Steps

### Step 1: Register the alias

In `slm_rl/games/mastermind/env.py`, after the class definition, add:

```python
# curriculum entry point: same engine, 3x4 = 64-code space (see plan 006)
register_game("mastermind-easy")(MastermindGame)
```

Make `eval_suite()` honest for both names: it cannot know its config, so keep
`game="mastermind"` but confirm nothing consumes `EvalSuite.game` for routing
(`grep -rn "suite.game\|\.game ==" slm_rl/` — if something routes on it,
STOP and report).

**Verify**: `uv run python -c "from slm_rl.games.registry import get_game; import slm_rl.games.mastermind.env; print(get_game('mastermind-easy').__name__)"` → `MastermindGame`.

### Step 2: Ship the config

`configs/games/mastermind-easy.yaml`, copying the exemplar with:
`max_turns: 10`, `eval_episodes: 500`, `eval_seeds_start: 10000`, same
monitor block, `extra: {code_length: 3, num_colors: 4, allow_duplicates: true}`.

**Verify**: `uv run python -c "from slm_rl.config.loader import load_game_config; c=load_game_config('mastermind-easy'); print(c.name, c.extra)"` → `mastermind-easy {'code_length': 3, 'num_colors': 4, 'allow_duplicates': True}`.

### Step 3: Relax the two name checks

- `grpo_export.py`: `if game_cfg.name != "mastermind":` →
  `if not game_cfg.name.startswith("mastermind"):`
- `teachers/__init__.py`: both `== "mastermind"` →
  `.startswith("mastermind")` (the solver and pruner read
  colors/length/dup from the instantiated game, so they generalize as-is).

**Verify**: `uv run pytest tests/test_grpo_export.py tests/test_solver.py tests/test_pruner.py -q` → pass.

### Step 4: Tests + smoke

New `tests/test_curriculum.py`:
- `get_game("mastermind-easy")` resolves; a reset game has
  `len(game._actions) == 64`, `colors == "RGBY"`.
- Solver teacher on the easy config wins ≥ 0.98 over 50 seeded episodes via
  `EpisodeRunner` (pattern: `tests/test_solver.py::test_solver_wins_and_is_deterministic`).
- `export_grpo_dataset` accepts the easy config (no NotImplementedError) on a
  2-record synthetic JSONL (pattern: `tests/test_grpo_export.py`).

Run the CPU smoke: `uv run slm-rl rollout --game mastermind-easy --agent solver --episodes 10`.

**Verify**: `uv run pytest -q` → all pass; smoke prints `win rate: 10/10`.

## Test plan

Covered in Step 4.

## Done criteria

- [ ] `uv run pytest -q` exits 0 (including new curriculum tests)
- [ ] Smoke command wins 10/10 on mastermind-easy
- [ ] `grep -rn '== "mastermind"' slm_rl/ | grep -v test` → no matches remain in `teachers/__init__.py` or `grpo_export.py`
- [ ] `configs/games/mastermind.yaml` unchanged (`git diff --stat -- configs/games/mastermind.yaml` empty)
- [ ] `plans/README.md` status row updated

## STOP conditions

- `register_game("x")(Cls)` double-registration raises (registry forbids
  aliases) — report; the fallback (a subclass) needs an operator decision.
- Anything routes on `EvalSuite.game` (Step 1 grep hits) — report before
  changing eval semantics.
- Eval seeds for the easy game would collide with training seeds in a way the
  standard game avoids (training seeds are small ints; eval starts at 10,000 —
  if you find otherwise, report).

## Maintenance notes

- Runs on the easy game are a *different benchmark* — never compare its
  win rates to standard-game numbers in the same table without labels.
- Deferred by design: automatic ramping (promote N times on easy → switch to
  standard). Do manually first: `slm-rl evolve --game mastermind-easy ...`,
  then re-`evolve --game mastermind` from the promoted adapter. Automate only
  once the manual ramp has worked once.
- The 4-color alphabet is `RGBY` (first 4 of `ALL_COLORS = "RGBYOP"`); the
  system prompt renders color names automatically.
