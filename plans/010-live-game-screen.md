# Plan 010: Live game screen in the web UI — deterministic frame replay for Atari runs

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving on. If
> any STOP condition occurs, stop and report — do not improvise. Commit in
> the worktree per the git workflow. Skip updating `plans/README.md` (the
> reviewer maintains the index).
>
> **Drift check (run first)**: `git diff --stat b5de903..HEAD -- slm_rl/webui/ tests/test_webui*.py` → must be empty. NOTE: another executor is concurrently editing `slm_rl/bridges/gym_adapter.py` and `slm_rl/teachers/` — those files are strictly OUT of your scope; do not read state from them, and expect them to differ from main by the time you finish. Your work must not touch them.

## Status

- **Priority**: P1 (user-requested: "can we see it playing the game live")
- **Effort**: M
- **Risk**: LOW (webui-only; read-only observer invariant unchanged)
- **Depends on**: 007 (webui) + 008 (Space Invaders), both landed
- **Category**: product (observability)
- **Planned at**: commit `b5de903`, 2026-07-11

## Why this matters

The web UI (plan 007) streams decisions as text/action cards. For Atari the
user wants to see the actual game screen. Recording frames during rollout
would bloat records and touch the play path — but plan 008 empirically
established that ALE with `repeat_action_probability=0.0` is **byte-
deterministic given a seed and action script**, and every record already
carries `(seed, parsed_action)` per step. So the viewer can *re-simulate*
any episode locally and render real frames — a pure read-only observer
(CODING_GUIDELINE invariant 5), zero change to rollout/training, and it
works both for finished episodes and near-live (frames follow the record
stream as the model plays).

## Current state

- `slm_rl/webui/tailer.py` — `iter_run_records(run_dir, poll_interval,
  stop)` yields record dicts; `to_event(rec)` builds the wire payload
  (includes `episode_id`, `generation`, `seed`, `parsed_action`,
  `terminated`, `truncated`).
- `slm_rl/webui/server.py` — `_make_handler(run_dir)`; routes: `/` (PAGE),
  `/events` (SSE via bounded queue + `_feed`), else 404. `serve(run_dir,
  host, port)`.
- `slm_rl/webui/page.py` — `PAGE` string; episode cards built in JS from
  `/events`; cards know their `episode_id` and `generation`.
- Rollout records live at `run_dir/generations/gen_NNN/rollouts/*.jsonl`;
  each line has `game` (e.g. `"space-invaders"`), `seed`, `step_idx`,
  `parsed_action` (an ALE meaning string like `"LEFT"` for atari games).
- `slm_rl/games/atari/space_invaders.py` / `configs/games/space-invaders.yaml`
  — the env is `gymnasium.make(extra["env_id"], obs_type="ram",
  frameskip=4, repeat_action_probability=0.0)`; each decision applies its
  action `extra["action_repeat"]` (=3) consecutive `env.step` calls with
  early break on terminated/truncated. Confirm the exact `gym.make` kwargs
  by reading `slm_rl/bridges/gym_adapter.py` on MAIN (read-only — remember
  it is out of scope to edit and may be mid-edit by another agent; read the
  version at commit `b5de903` via `git show b5de903:slm_rl/bridges/gym_adapter.py`
  to be safe).
- `slm_rl/config/loader.py` — `load_game_config(name)` gives `max_turns`
  and `extra` for a game name.
- webui is stdlib-only for its core; the atari extra (gymnasium/ale-py,
  which bring numpy) is required only for frame replay — imports must be
  lazy inside the replay module, and the endpoint must degrade gracefully
  (HTTP 501 + plain message) when the extra is absent.
- `zlib` is stdlib: a minimal PNG encoder (8-bit RGB, no filter) is ~30
  lines — no PIL, no new dependencies.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Env | `uv sync --extra cuda --extra dev --extra atari` | ok (once) |
| Tests | `uv run --no-sync pytest -q` | all pass |
| Focused | `uv run --no-sync pytest tests/test_webui_replay.py -q` | all pass |

## Scope

**In scope**:
- `slm_rl/webui/replay.py` (new), `slm_rl/webui/png.py` (new, minimal encoder)
- `slm_rl/webui/server.py` (one new route), `slm_rl/webui/page.py` (screen panel)
- `tests/test_webui_replay.py` (new)
- `docs/ARCHITECTURE.md` (extend the existing `slm-rl watch` subsection by 2–3 sentences)

**Out of scope** (do NOT touch):
- `slm_rl/bridges/gym_adapter.py`, `slm_rl/games/`, `slm_rl/teachers/`
  (concurrent executor + not needed — replay builds its own env directly
  from the game config).
- `slm_rl/rollout/`, `slm_rl/eval/`, `slm_rl/orchestrator/`, `pyproject.toml`.
- `slm_rl/webui/tailer.py` — reuse `iter_run_records` as-is.

## Git workflow

- Conventional commits + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer. No push unless instructed.

## Steps

### Step 1: PNG encoder

`slm_rl/webui/png.py`: `encode_rgb(pixels: bytes, width: int, height: int)
-> bytes` — PNG signature + IHDR + IDAT (zlib, filter byte 0 per scanline) +
IEND, stdlib only (`zlib`, `struct`). Unit-test it against a known 2×2
image (hand-compute the bytes or round-trip via a decoder ONLY if one is
available without new deps — otherwise assert signature/chunk structure and
that common browsers' requirements hold: correct IHDR dims, CRCs valid).

### Step 2: The replayer

`slm_rl/webui/replay.py`:

- `replay_frames(run_dir: Path, episode_id: str, stop=None) -> Iterator[bytes]`
  — PNG frames for one episode:
  1. Scan the run's rollout JSONLs (reuse `iter_run_records` with an
     immediate-stop event for catch-up, or a targeted file scan) for
     records with that `episode_id`, ordered by `step_idx`; note `game`
     and `seed` from the first record.
  2. `load_game_config(game)`; if `extra` has no `env_id` → raise
     `ReplayUnavailable("no visual replay for this game")` (mastermind
     etc.).
  3. Lazy-import gymnasium/ale_py; build the env EXACTLY as the adapter
     does (same env_id/frameskip/repeat_action_probability) but with
     `render_mode="rgb_array"`; `reset(seed=seed)`.
  4. Per record: map `parsed_action` → index via
     `env.unwrapped.get_action_meanings()`; step it `action_repeat` times
     (early break on terminated/truncated), calling `env.render()` after
     each step and yielding each frame as PNG (`encode_rgb`). Unknown
     action id (old/corrupt record) → stop replay cleanly, don't raise.
  5. **Near-live follow**: after consuming all currently-available records
     for the episode, poll for new ones (0.5s) until a record with
     `terminated`/`truncated` true was seen or `stop` is set.
- Frames are 210×160 RGB numpy arrays; convert via `.tobytes()`.

### Step 3: Server route + page panel

- `server.py`: `GET /frames?episode=<id>` → `multipart/x-mixed-replace`
  (MJPEG-style but with PNG parts) OR SSE with base64 data-URIs — pick
  multipart (native `<img src="/frames?...">`, no JS decoding, lower
  overhead). Same bounded-backpressure pattern as `/events` (feeder thread
  + `queue.Queue(maxsize=8)` — frames are bigger; drop-oldest is fine for
  a live screen: use a small deque semantics or put with timeout and skip).
  Client disconnect → stop event → env closed (`finally: env.close()`).
  `ReplayUnavailable` or missing atari extra → 501 with the message.
  Cap: at most 4 concurrent replay streams (a module-level semaphore);
  505th client gets 503 — log nothing, it's a local tool.
- `page.py`: each episode card gets a "▶ watch" button; clicking opens a
  fixed panel with `<img>` pointed at `/frames?episode=<id>` (and a close
  button that clears `src` — this is what ends the stream server-side).
  Throttle: the server yields frames at a natural pace — insert
  `time.sleep(1/30)` between yielded frames in the route (not in
  replay.py, keep the generator pure).

### Step 4: Tests

`tests/test_webui_replay.py` (`pytest.importorskip("ale_py")` for the
env-dependent ones; PNG tests always run):

1. PNG: valid signature, IHDR dims, zlib-decompressible IDAT for a 2×2 image.
2. Replay determinism: write a synthetic 3-record JSONL for a real
   space-invaders episode (drive a real env yourself in the test to get
   valid actions/seed), then `replay_frames` yields ≥ 3× action_repeat
   frames minus early-breaks; calling it twice yields byte-identical frames.
3. `ReplayUnavailable` for a mastermind record set.
4. HTTP: `/frames?episode=missing` → 404 or clean empty end (pick one,
   assert it); missing-extra path → 501 (monkeypatch the import to fail).
5. Follow-mode stops: with `stop` set after the first frame, the generator
   terminates and the env is closed (assert via a closed flag/monkeypatched
   close).

**Verify**: `uv run --no-sync pytest -q` → all pass.

### Step 5: Manual smoke + doc

With the real run data on this machine (`runs/spaceinv-350m` — READ ONLY):
serve on a scratch port, `curl -s "localhost:<port>/frames?episode=<a real
episode_id from the jsonl>" | head -c 2000 | grep -a PNG` → PNG bytes
appear. Extend the ARCHITECTURE.md watch subsection: frame replay is
re-simulation from records (determinism argument), atari-only, read-only.

## Test plan

Covered in Step 4; the determinism test (2) is the core correctness claim.

## Done criteria

- [ ] `uv run --no-sync pytest -q` exits 0
- [ ] `git diff --stat -- slm_rl/bridges/ slm_rl/games/ slm_rl/teachers/ slm_rl/rollout/ slm_rl/orchestrator/ pyproject.toml` → empty
- [ ] `grep -rn "import gymnasium\|import ale_py" slm_rl/webui/ | grep -v "def \|lazy"` shows imports only inside function bodies
- [ ] Manual smoke: PNG bytes stream for a real spaceinv-350m episode
- [ ] `plans/README.md` status row updated (by reviewer)

## STOP conditions

- The adapter's env construction kwargs (read from `git show b5de903:...`)
  can't be mirrored without importing from `slm_rl.bridges` (coupling you
  must not create — copy the 3 kwargs with a comment instead; if they're
  more complex than env_id/frameskip/repeat_action_probability/action_repeat,
  report).
- Re-simulated frames are NOT deterministic across two replays of the same
  records (breaks the whole design — report with evidence).
- You need to modify `tailer.py` or the rollout writer to find episodes
  efficiently (out of scope — a linear scan of the run's JSONLs is fine at
  this scale).

## Maintenance notes

- Replay correctness depends on the adapter's env kwargs staying in sync
  with replay.py's copies — a comment in both places (only yours is
  editable now; note the adapter-side comment as a follow-up for the
  reviewer) and the determinism test guard it.
- When plan 011 (vision-model play) lands, frames become model *inputs* —
  this module's env-rebuild logic is the natural place to share; revisit
  then, don't pre-build.
- Non-Atari games get 501 by design; a Mastermind board animation would be
  a separate, pure-JS feature (records already carry everything needed).
