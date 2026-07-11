# Plan 001: Adopt the teacher warm-start unconditionally as the RL initialization

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat cd2b4f4..HEAD -- slm_rl/orchestrator/generation.py slm_rl/cli.py tests/test_generation.py docs/DECISIONS.md`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: tech-debt (training-pipeline correctness)
- **Planned at**: commit `cd2b4f4`, 2026-07-11

## Why this matters

SLM-RL's teacher warm start (`slm-rl evolve --warm-start`) currently runs as a
*gated candidate*: generation 1 is trained by SFT on expert-teacher episodes,
then must beat the raw-base champion's eval score by the promotion margin, or
its adapter is discarded. Measured twice (runs `hybrid-350m`, `hybrid-350m-v2`,
July 2026): the warm start dramatically improved behavior (invalid actions
35%→0%, doom-loop interventions 2.0→1.5) but its win rate (0.4–1.6%) sat at
the margin, was rejected both times, and **every subsequent RL generation
restarted from the raw base model and regressed**. Standard post-training
practice (SFT stage → RL stage, as in InstructGPT-style pipelines) treats SFT
as *initialization*, not as a competitor. After this plan, the warm-start
generation is adopted unconditionally as the champion so RL generations
always build on it; its eval is still recorded honestly.

## Current state

- `slm_rl/orchestrator/generation.py` — `GenerationRunner.run_generation(generation, teacher=False)`
  runs ROLLOUT→DATASET→TRAIN→EVAL→GATE. The teacher branch (lines ~110–125)
  swaps in the solver agent and forces `reject_sft`, but the gate section
  (lines ~160–175) treats the result like any candidate:

  ```python
  # generation.py (gate section, abbreviated)
  else:
      cand_metrics = self._eval(result.adapter_path)
      promote, reason = self.gate.decide(champ_metrics, cand_metrics)

  if promote:
      self.registry.promote(generation, reason)
      self._write_json(self.paths.generation(generation) / "eval" / "results.json", cand_metrics)
      # remediation is a crutch, not a new baseline: a promotion clears it
      self.cfg.train.learning_rate = self._orig_lr
      self.cfg.train.entropy_bonus = self._orig_entropy_bonus
  else:
      self.registry.reject(generation, reason)
  ```

- `slm_rl/cli.py` — `evolve` command has the warm-start branch:

  ```python
  if warm_start:
      if start == 1:
          m = runner.run_generation(1, teacher=True)
  ```

- `tests/test_generation.py` — `test_warm_start_teacher_generation` currently
  asserts `m["gate"]["promoted"] is True` with a scripted candidate eval of
  0.50 vs champion 0.00 (it only passes because the scripted margin is large).
  Helper `make_runner(...)` monkeypatches `runner._eval` with a scripted
  metrics iterator; `FakeStrategy` produces a dummy adapter.

- Registry semantics (`slm_rl/orchestrator/registry.py`): `promote(gen, reason)`
  moves the champion pointer; `reject` records history only. `EvalGate.decide`
  (`slm_rl/eval/gate.py`) is pure and takes `(champion_metrics, candidate_metrics)`.

- Repo conventions: pydantic config models in `slm_rl/config/schema.py`;
  deliberate shortcuts marked with `# ponytail:` comments; tests use
  fakes + monkeypatched lazy factories (see `tests/test_generation.py`).

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Tests (all) | `uv run pytest -q` | all pass (90+ tests) |
| Tests (this area) | `uv run pytest tests/test_generation.py -q` | all pass |

## Scope

**In scope** (the only files you should modify):
- `slm_rl/orchestrator/generation.py`
- `tests/test_generation.py`
- `docs/DECISIONS.md` (append one decision entry)

**Out of scope** (do NOT touch, even though they look related):
- `slm_rl/eval/gate.py` — gate logic is unchanged; RL generations still gate normally.
- `slm_rl/orchestrator/registry.py` — promote/reject semantics unchanged.
- `slm_rl/cli.py` — the `--warm-start` flag and branch already work; no change needed.
- `configs/*` — no new config knobs for this.

## Git workflow

- Branch: work on the current branch unless the operator says otherwise.
- Commit style: conventional commits, e.g. `feat: adopt teacher warm-start unconditionally as RL init` with trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` (match `git log --oneline -5`).
- Do NOT push unless the operator instructed it.

## Steps

### Step 1: Bypass the gate for teacher generations

In `slm_rl/orchestrator/generation.py`, in `run_generation`, change the
EVAL+GATE section so that when `teacher=True` **and** training produced an
adapter (`result.adapter_path is not None` and not
`result.metrics.get("entropy_collapsed")`), the candidate is still evaluated
(the eval numbers must stay honest) but the gate is skipped:

```python
elif teacher:
    cand_metrics = self._eval(result.adapter_path)
    promote = True
    reason = "teacher warm-start adopted as RL initialization (not gated)"
else:
    cand_metrics = self._eval(result.adapter_path)
    promote, reason = self.gate.decide(champ_metrics, cand_metrics)
```

Keep the existing `entropy_collapsed` and `adapter_path is None` branches
ABOVE this — a collapsed or empty teacher generation must still be rejected
(those two branches already set `promote=False` and reuse champion metrics).

**Verify**: `uv run pytest tests/test_generation.py -q` → existing tests still pass
(the warm-start test asserted promotion with a big margin, so it stays green).

### Step 2: Tighten the warm-start test to prove the gate is bypassed

In `tests/test_generation.py`, edit `test_warm_start_teacher_generation` to
script a candidate eval that would FAIL the gate (e.g. `champ_primary=0.10`,
`cand_primary=0.10` — zero improvement), and assert:

- `m["gate"]["promoted"] is True`
- `m["gate"]["reason"]` contains `"not gated"`
- `runner.registry.champion == 1`
- `(runner.paths.generation(1) / "eval" / "results.json").exists()` (honest eval recorded)

Also add a companion test `test_warm_start_collapse_still_rejects`: build the
runner with `collapse=True` (existing `make_runner` parameter) and
`teacher_overrides={"warmstart_episodes": 3}`, run
`runner.run_generation(1, teacher=True)`, assert `promoted is False` and
`runner.registry.champion == 0`.

**Verify**: `uv run pytest tests/test_generation.py -q` → all pass including the 2 new/changed tests.

### Step 3: Record the decision

Append to `docs/DECISIONS.md` a new entry `## D12. SFT warm-start is initialization, not a candidate`:
state that teacher-distilled SFT generations are adopted unconditionally as
the RL init (eval recorded, gate bypassed), cite the July 2026 evidence (two
runs where a 35%→0% invalid-rate improvement was discarded over a noisy
win-rate margin and subsequent RL generations regressed), and note the
revisit condition: if a warm start ever *degrades* invalid/intervention rates
vs the raw base, re-introduce a behavioral (not win-rate) gate for it.

**Verify**: `grep -c "D12" docs/DECISIONS.md` → `>= 1`.

## Test plan

- Modified: `test_warm_start_teacher_generation` (gate bypass with a
  would-fail margin — this is the regression test for the whole plan).
- New: `test_warm_start_collapse_still_rejects` (entropy collapse still rejects).
- Pattern to follow: existing tests in `tests/test_generation.py` using
  `make_runner(...)` + scripted `_eval`.
- Verification: `uv run pytest -q` → all pass.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `uv run pytest -q` exits 0
- [ ] `grep -n "not gated" slm_rl/orchestrator/generation.py` returns ≥1 match
- [ ] `grep -n "D12" docs/DECISIONS.md` returns ≥1 match
- [ ] `git status --short` shows no modified files outside the in-scope list
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- The gate section of `run_generation` no longer matches the excerpt above
  (upstream drift).
- Any test outside `tests/test_generation.py` starts failing after your change
  (the bypass leaked into the non-teacher path).
- You find that `teacher=True` generations no longer force `reject_sft`
  (that invariant is assumed here and owned by the existing code).

## Maintenance notes

- Plan 004 (replay) also edits `run_generation`; execute this plan first to
  avoid conflicts.
- Reviewer should scrutinize: the non-teacher path must be byte-identical in
  behavior — only `teacher=True` bypasses the gate.
- Deferred deliberately: a behavioral gate for warm starts (invalid-rate /
  intervention-rate regression check). Add only if a teacher stage ever makes
  behavior worse (see D12 revisit condition).
