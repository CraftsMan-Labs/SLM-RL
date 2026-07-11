# Plan 002: Make the teacher verbalize its deduction (process supervision) and put strategy in the system prompt

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat cd2b4f4..HEAD -- slm_rl/teachers/mastermind_solver.py slm_rl/datagen/sft_export.py slm_rl/games/mastermind/env.py tests/test_solver.py tests/test_sft_export.py`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED (changes training-data content; win-rate effect must be measured, not assumed)
- **Depends on**: none (001 recommended first so the resulting SFT base is actually adopted)
- **Category**: direction (training-signal quality)
- **Planned at**: commit `cd2b4f4`, 2026-07-11

## Why this matters

The teacher warm start distills (game state → bare `ACTION: RGBY`) pairs. The
solver's actual skill — filtering the 1,296-code space against feedback —
happens *before* it speaks and never appears in the training tokens, so SFT
transferred only surface form. Measured result (July 2026, runs
`hybrid-350m*`): invalid actions 35%→0% but win rate ≈ legal-random (~0.9%
theoretical; 0.4–1.6% observed). Rationale distillation (STaR/Orca-style)
puts the *procedure* in the tokens: the teacher states what the feedback
eliminates and how many candidates remain, then acts. Paired with a
strategy-bearing system prompt, SFT becomes process supervision instead of
style copying. This is the highest-leverage change for the 350M tier and
also improves the 1.2B warm start.

## Current state

- `slm_rl/teachers/mastermind_solver.py` — `SolverAgent.act()` returns a bare
  action line:

  ```python
  guess = self._rng.choice(pool)
  return ActionDecision(
      action=by_id[guess],
      raw_completion=f"ACTION: {guess}",
      prompt_messages=build_messages(self.system_prompt, obs),
  )
  ```

  It already computes `cands = consistent_candidates(...)` (the full
  consistent set) and intersects with the menu into `pool` — the numbers a
  rationale needs are already in local scope.

- `slm_rl/datagen/sft_export.py` — `export_sft_dataset` (lines 102–107)
  **discards the record's raw completion** and emits a canonical answer:

  ```python
  row = {
      "prompt": prompt,
      "completion": [
          {"role": "assistant", "content": f"ACTION: {rec['parsed_action']}"}
      ],
  }
  ```

  so any rationale in the record would be dropped. Records carry `model_id`
  (e.g. `"teacher:mastermind_solver"`) and `completion` (the raw text) —
  see `slm_rl/datagen/schema.py`.

- `slm_rl/games/mastermind/env.py` — `system_prompt()` states the rules and
  "Never repeat a guess" but contains **no strategy**:

  ```python
  return (
      "You are playing Mastermind. A secret code of "
      f"{self.code_length} colors is hidden (colors: {color_list}; "
      ...
      "narrow down the code. Never repeat a guess — it gives no new "
      "information."
  )
  ```

- Parsing contract (`slm_rl/agents/llm_agent.py`): `extract_action_token`
  takes the **last** `ACTION:` line, so a rationale placed *before* the final
  `ACTION:` line parses fine — as long as the rationale itself never contains
  the string `ACTION:`.

- Budget rule (docs/ARCHITECTURE.md, "The 8GB principle"): whole conversation
  ≤ 2048 tokens; completions capped at `train.max_completion_tokens: 256`.
  Prompting is stateless per turn (completions are never fed back into later
  prompts), so a rationale costs completion tokens only.

- Tests: `tests/test_solver.py` asserts teacher records' prompts and that
  `export_sft_dataset` yields pairs; `tests/test_parser.py` has parsing
  goldens. There is a `tests/test_sft_export.py` — open it before editing to
  follow its fixture style.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Tests (all) | `uv run pytest -q` | all pass |
| Tests (this area) | `uv run pytest tests/test_solver.py tests/test_sft_export.py tests/test_parser.py -q` | all pass |

## Scope

**In scope**:
- `slm_rl/teachers/mastermind_solver.py`
- `slm_rl/datagen/sft_export.py`
- `slm_rl/games/mastermind/env.py` (system_prompt only)
- `tests/test_solver.py`, `tests/test_sft_export.py`

**Out of scope** (do NOT touch):
- `slm_rl/agents/llm_agent.py` — the `"Think briefly if needed, then end with
  one line: ACTION: <your move>"` instruction and `build_messages` stay
  exactly as-is (several parser goldens assert this string).
- `slm_rl/training/grpo.py` and `slm_rl/datagen/grpo_export.py` — GRPO rewards
  parse the last ACTION line and are rationale-agnostic; no change needed.
- `slm_rl/teachers/pruner.py`.

## Git workflow

- Commit style: conventional commits + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer (match `git log`).
- Do NOT push unless instructed.

## Steps

### Step 1: Emit a compact rationale from the solver

In `SolverAgent.act()`, build a 1–2 line rationale before the action line.
Use only numbers already in scope; deterministic; NEVER contains the string
`ACTION:` except in the final line. Target shape:

```python
total = len(consistent_candidates(self.colors, self.code_length, self.dup_ok, []))
# ... cands/pool as today ...
if hist:
    rationale = (
        f"The feedback so far eliminates {total - len(cands)} of {total} possible codes; "
        f"{len(cands)} candidates remain consistent with every (exact, partial) result. "
        f"{guess} is one of them."
    )
else:
    rationale = f"No feedback yet: all {total} codes are possible. {guess} probes {len(set(guess))} distinct colors."
completion = f"{rationale}\nACTION: {guess}"
```

Notes: compute `total` once in `__init__` (it's constant per config), not per
turn. `hist` is `obs.metadata.get("history", [])`.

**Verify**: `uv run pytest tests/test_solver.py -q` → currently-passing tests
may fail on completion-shape assertions — fix them in Step 4, but the win-rate
and determinism tests must still pass now (`-k "wins or deterministic"`).

### Step 2: Preserve teacher rationales through SFT export

In `export_sft_dataset` (`slm_rl/datagen/sft_export.py`), when a record's
`model_id` starts with `"teacher:"`, emit the record's raw `completion`
verbatim instead of the rebuilt `ACTION: {parsed_action}`:

```python
is_teacher = str(rec.get("model_id", "")).startswith("teacher:")
content = rec["completion"] if is_teacher else f"ACTION: {rec['parsed_action']}"
```

Do NOT use raw completions for non-teacher records — model-generated text can
contain retry junk; the canonical rebuild stays for those. Add a
`# ponytail:` comment saying exactly that.

**Verify**: `uv run pytest tests/test_sft_export.py -q` → pass.

### Step 3: Add strategy to the Mastermind system prompt

In `MastermindGame.system_prompt()`, append two sentences of strategy (keep
the whole prompt under ~120 tokens; count words ≈ tokens×0.75):

> "Strategy: every feedback eliminates codes — a good guess is consistent
> with ALL previous feedback (it would have produced exactly those exact and
> partial counts). Before guessing, state briefly what the feedback rules
> out, then choose a consistent code."

**Verify**: `uv run pytest -q` → all pass (no test asserts the full prompt
text; if one does, treat as drift → STOP).

### Step 4: Update tests

- `tests/test_solver.py`: update the record-shape test to assert the teacher
  completion (a) contains at least one digit (the candidate count), (b) ends
  with `ACTION: <code>` as the last line, and (c) `extract_action_token`
  (import from `slm_rl.agents.llm_agent`) still recovers the exact guess.
  Add: rationale never contains `"ACTION:"` before the final line
  (`completion.count("ACTION:") == 1`).
- `tests/test_sft_export.py`: add a test with two synthetic records — one
  `model_id="teacher:x"` with `completion="Because...\nACTION: RRRR"`, one
  `model_id="llm"` with junky completion — assert the teacher pair keeps the
  rationale verbatim and the llm pair is rebuilt to `ACTION: <parsed_action>`.
  Follow the existing fixture style in that file.

**Verify**: `uv run pytest -q` → all pass.

## Test plan

Covered in Step 4. The two regression anchors: (1) rationale never breaks the
`ACTION:` parsing contract; (2) teacher-vs-LLM export branching.

## Done criteria

- [ ] `uv run pytest -q` exits 0
- [ ] `python3 -c "from slm_rl.config.loader import load_game_config; from slm_rl.teachers import make_teacher; from slm_rl.games.mastermind.env import MastermindGame; cfg=load_game_config('mastermind'); a,_=make_teacher(cfg,seed=0); g=MastermindGame(cfg); obs=g.reset(3); d=a.act(obs,[]); print(d.raw_completion)"` prints a rationale line followed by `ACTION: <4 letters>`
- [ ] `grep -n "teacher:" slm_rl/datagen/sft_export.py` returns ≥1 match
- [ ] `grep -n "Strategy:" slm_rl/games/mastermind/env.py` returns ≥1 match
- [ ] No files outside the in-scope list modified (`git status --short`)
- [ ] `plans/README.md` status row updated

## STOP conditions

- `export_sft_dataset` no longer rebuilds completions from `parsed_action`
  (drift — someone already changed the export contract).
- Any parser golden in `tests/test_parser.py` fails (your rationale leaked
  into the parsing contract).
- The rationale pushes teacher completions past ~80 tokens (violates the
  8GB completion budget philosophy — shorten, and if you can't, report).

## Maintenance notes

- Changing prompt/data content breaks cross-run comparability: eval numbers
  from runs before this plan are not comparable to runs after. Start new
  run-ids after landing.
- If a future game adds a teacher, its rationale must follow the same
  contract: plain text, single `ACTION:` occurrence, last line.
- Deferred: verbalizing *which specific prior guess* eliminated the most
  candidates (richer but longer); only pursue if the compact rationale
  measurably moves win rate.
