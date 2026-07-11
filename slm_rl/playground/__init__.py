"""`slm-rl playground`: a stdlib-only workshop UI for tweaking reward
functions and run knobs, launching quick CPU experiments (teacher/random
rollouts), and comparing them on a scoreboard before committing to a real
`evolve` run.

Read/write surface: unlike `slm_rl/webui/` (a pure read-only observer, see
CODING_GUIDELINE invariant 5), this package DOES write to disk — but only
under `runs/playground/<home>/<name>/` (materialized experiment config
dirs + reward hook files it creates itself). It never touches
`slm_rl/training/`, `slm_rl/datagen/`, `slm_rl/eval/`,
`slm_rl/orchestrator/`, `slm_rl/teachers/`, or repo `configs/*.yaml`.

Stdlib-only (CODING_GUIDELINE 8GB rule): no `gymnasium`/`ale_py`/`numpy`
imports anywhere in this package. Experiments run in a subprocess
(`python -m slm_rl.cli rollout/evolve`), which is where those heavy/optional
imports actually happen.

Security model: the reward-code tab executes attendee-written Python
(`shape_reward(ctx) -> float`) via `importlib`. This is a local workshop
tool run on the attendee's own machine — the same trust model as them
editing the repo directly. Do not expose this server beyond localhost.
"""

from __future__ import annotations
