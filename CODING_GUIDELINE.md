# SLM-RL Coding Guideline

Audience: any engineer or agent writing code in this repo — especially plan
executors. Read this fully before your first edit. When a plan and this
guideline conflict, the plan wins (it was written with more context); say so
in your report.

## 1. Non-negotiable invariants

These are design decisions (see `docs/DECISIONS.md`), not preferences.
Violating one fails review regardless of test results.

1. **The 8GB rule.** The full loop (play → datagen → train → eval) must run
   on an 8GB-RAM machine. Core `slm_rl` must import and run with NO optional
   extras installed — heavy imports (`torch`, `transformers`, `pyarrow`,
   `streamlit`, ...) live behind lazy imports inside functions, never at
   module top level of core packages. Stream and chunk file IO; never load a
   whole dataset into memory when a generator works.
2. **Eval-gate purity.** Steering (teachers, pruners, monitors' interventions)
   must never be counted as model improvement. The gate eval is always
   LLM-only, unpruned. Anything that could leak assistance into
   `gate.decide` inputs is a bug, even if metrics look better.
3. **Games stay ML-free.** Nothing under `slm_rl/games/` may import from
   `training/`, `inference/`, `datagen/`, or `teachers/`. Games expose the
   `Game` ABC (Gymnasium-like contract) and nothing else.
4. **Determinism.** Every stochastic choice takes an explicit seed
   (`random.Random(seed)`, never module-level `random.*` or wall-clock).
   Derived seeds are arithmetic (`seed * 10_007 + turn`), never tuples or
   hashes of objects. Same seed → byte-identical decisions.
5. **Read-only observers stay read-only.** Dashboards, watchers, and
   exporters never mutate `runs/` — open files for reading only.

## 2. Style — match the neighborhood

- Python 3.13, 4-space indent, type hints on all public signatures,
  `from __future__ import annotations` at module top.
- Config: pydantic models in `slm_rl/config/schema.py`, every field mirrored
  in `configs/default.yaml` **with the same default**. New knobs default to
  "current behavior unchanged" so existing runs reproduce identically.
- Dataclasses for data records (`slm_rl/datagen/schema.py` is the exemplar);
  bump `schema_version` on breaking record changes and keep readers
  backwards-compatible.
- Deliberate simplifications carry a `# ponytail: <what the real version
  would be>` comment. Use them; they are how future readers find shortcuts.
- Comments state constraints the code can't show ("never empty by
  construction because score_guess(g,g)=(n,0)"), not narration of the next
  line.
- Errors: fail loudly in the orchestrator (a bad run must die visibly);
  degrade gracefully inside rollout (a bad completion becomes
  `parse_status="fallback_random"`, never a crash).
- CLI: typer commands in `slm_rl/cli.py`; flags are `--kebab-case` with
  `--no-` negations for booleans.

## 3. Testing standards

- `uv run pytest -q` green is the floor, not the goal. Every behavior change
  ships with a test that FAILS on the old code.
- Use the repo's fakes: `FakeBackend`/`FakeStrategy`/`make_runner` in
  `tests/test_generation.py`, scripted backends in `tests/test_parser.py`.
  Never load a real model or touch the GPU in tests.
- Golden values are hand-computed and the computation shown in a comment.
  For Mastermind, verify against `score_guess` by hand on a 2-peg/2-color
  case first — intuition about consistency is wrong more often than it
  feels (measured: it burned us on "GG leaves only {RG}").
- Boundary tests pick values clearly away from float artifacts (0.305, not
  0.31 vs 0.31000000000000004).
- A test that cannot fail (asserts on its own fixture, tautological mocks)
  is worse than no test — it certifies nothing and blocks refactors.

## 4. Second- and third-order consequences — think before you type

Before finishing any change, answer these in your report's NOTES:

1. **Reproducibility:** does this change make old runs incomparable to new
   ones (prompt text, reward scale, selection logic)? If yes, say so —
   operators must start new run-ids.
2. **The other trainer:** reject_sft and GRPO share exporters and records.
   A change tuned for one — does it alter what the other sees?
3. **Resume:** `slm-rl evolve` resumes mid-run. Does your change behave when
   generation N−1 ran on the OLD code (missing fields, different paths)?
   Readers tolerate missing keys (`rec.get(...)`), writers always write them.
4. **The 350M floor:** will this hold up on the weakest tier (CPU, 8GB,
   1000-episode warm starts), not just the CUDA box?
5. **Silent caps:** if you bound anything (top-N, truncation, sampling),
   log what was dropped. Silent truncation reads as "covered everything".

## 5. Playground / workshop (plan 026+)

Attendee path is **browser-only**. CLI/`docker compose` bootstrap instructors;
do not add new day-of CLI workflows for attendees.

1. **UI-first multi-game.** One playground process; game chosen in UI at
   experiment create (`get_game(name)`). Persist `game` on each experiment;
   scoreboard shows a game column + filters. CLI `--game` is optional override.
2. **Experiment, not project.** Flat `runs/playground/<name>/`. New name =
   new row. Same container may hold exps for many games.
3. **Single-flight locks.** At most one quick + one evolve + one theater
   subprocess (plan 013). Surface HTTP 409 Busy clearly in UI — no queues.
4. **Official HF model IDs only.** Presets must be org-owned repos
   (`Qwen/…`, `LiquidAI/…`, `google/…`, `nvidia/…`). No community forks,
   no invented small Qwen3.6 IDs (those open weights are 27B / 35B-A3B only).
5. **Token hygiene (plan 021).** Profile token file mode 0600; API masks
   token; never put tokens in logs, argv, or commits. Token optional — no
   hard signup gate.
6. **Live log tailer is product.** `GET .../log` for rollout/evolve/theater
   is intentional (plan 026). Do not delete it in complexity-cut passes.
7. **Tournament = honor system.** No shared multi-user eval / ELO product
   unless a plan explicitly adds it.
8. **ALE games.** Same recipe as plan 016: verified RAM map docstring +
   renderer + yaml + heuristic teacher + tests. Unverifiable core RAM →
   STOP that game, NOTES, continue others. Games stay `reject_sft`-only
   unless a plan says otherwise.

## 6. Git workflow

- Conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`), one
  logical change per commit, trailer:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Never push. Never commit files outside your declared scope. Never commit
  `note.txt`, `runs/`, or anything under `.venv/`.
- Plan executors: one commit per plan phase when the plan lists phase
  commits; `uv run pytest` green before each commit.

## 7. Definition of done (self-review before reporting)

- [ ] Every done criterion of the plan re-run and observed passing — paste
      actual command output in your report, not "should pass".
- [ ] `git status --short` shows only in-scope files.
- [ ] Diff re-read top to bottom as a reviewer would: no debug prints, no
      commented-out code, no TODO without a ponytail, no unused imports.
- [ ] New knobs documented where they live (docstring + yaml comment).
- [ ] Invariants §§1 and §5 (playground) re-checked against your diff.
- [ ] Anything surprising, deviated, or judgment-called is in NOTES —
      an undocumented deviation is a review failure even when the code is
      right.

## 8. Subagent / plan-executor briefing

Before editing: read this file + the active `plans/0NN-*.md` fully. Honor
the plan's STOP rules. Prefer shortest diff. Report NOTES for any
judgment call. Do not push. Do not expand scope into out-of-scope tables.
