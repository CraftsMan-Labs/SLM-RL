# Plan 007: Live-play web UI — stream Mastermind games from a run in the browser

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat cd2b4f4..HEAD -- slm_rl/webui/ slm_rl/cli.py slm_rl/datagen/schema.py tests/test_webui.py`
> `slm_rl/webui/` and `tests/test_webui.py` must not exist yet. If
> `RolloutRecord` in `schema.py` changed materially vs the excerpt below, STOP.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: LOW (pure observer; zero interaction with training/eval paths)
- **Depends on**: none (visual payoff is bigger after 002 lands — rationales stream too)
- **Category**: product (observability)
- **Planned at**: commit `cd2b4f4`, 2026-07-11

## Why this matters

Today the only way to watch a model play is `tail -f` on a JSONL of dense
records. A browser view that streams each guess as a row of colored pegs —
with the model's raw completion (including teacher/model rationales once
plan 002 lands), per-step reward, parse status, and doom-loop flags — makes
run health legible at a glance and is the project's first product surface.
The rollout pipeline already streams one `RolloutRecord` JSON line per
decision to disk during play, so live streaming is a *tail, parse, push*
problem: no hooks into the runner, no risk to the training loop.

## Current state

- **Records**: `slm_rl/datagen/schema.py` defines `RolloutRecord` (dataclass,
  `schema_version = 1`) with fields including `run_id`, `generation`, `game`,
  `episode_id`, `step_idx`, `seed`, `model_id`, `prompt_messages`
  (full chat, heavy), `completion` (raw model text), `parsed_action`,
  `legal_actions`, `parse_status` (`ok | retry_ok | fallback_random`),
  `reward`, `cum_reward`, `terminated`, `truncated`,
  `outcome` (`win | loss | draw | score:<n>`, terminal steps only),
  `monitor_flags`, `timestamp`. One JSON object per line.

- **Layout**: `slm_rl/orchestrator/paths.py` — rollout JSONLs live at
  `runs/<run_id>/generations/gen_NNN/rollouts/*.jsonl` (e.g.
  `mastermind.jsonl`). Eval episodes are not recorded there; this UI shows
  rollout play only. New `gen_NNN` directories appear as a run progresses;
  the file is appended to while episodes play.

- **Existing dashboard**: `slm_rl/dashboard/app.py` is a Phase-4 *metrics*
  dashboard stub (Streamlit, `[dashboard]` extra, cross-generation curves).
  This plan is a different, dependency-free surface. Do not touch it.

- **Dependency policy**: `pyproject.toml` core deps are pydantic/pyyaml/
  typer/psutil only; comment: "the core package must import and run on an
  8GB machine with none of these installed." The web UI must therefore be
  **stdlib only** (`http.server`, `json`, `pathlib`, `threading`).

- **CLI**: `slm_rl/cli.py` is a typer app; `dashboard()` command exists at
  ~line 307. Commands use lazy imports inside the function body.

- **Mastermind colors**: codes are strings like `"RGBY"` over the alphabet
  `RGBYOP` (R=Red, G=Green, B=Blue, Y=Yellow, O=Orange, P=Purple).

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Tests | `uv run pytest -q` | all pass |
| Focused | `uv run pytest tests/test_webui.py -q` | all pass |
| Smoke | `uv run slm-rl watch --run <existing run id> --port 8777` then `curl -s localhost:8777/` | HTML containing "SLM-RL" |

## Scope

**In scope**:
- `slm_rl/webui/__init__.py`, `slm_rl/webui/tailer.py`, `slm_rl/webui/page.py`,
  `slm_rl/webui/server.py` (all new)
- `slm_rl/cli.py` (one new `watch` command)
- `tests/test_webui.py` (new)
- `docs/ARCHITECTURE.md` (one short subsection under the observability/
  dashboard area)

**Out of scope** (do NOT touch):
- `slm_rl/rollout/`, `slm_rl/eval/`, `slm_rl/orchestrator/` — the UI is an
  observer; no hooks, no callbacks, no extra writes from the play path.
- `slm_rl/dashboard/` — the metrics dashboard stub stays as-is.
- `pyproject.toml` — no new dependencies of any kind.

## Git workflow

- Conventional commits + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer. No push unless instructed.

## Steps

### Step 1: The tailer (pure logic, fully unit-testable)

`slm_rl/webui/tailer.py`:

- `iter_run_records(run_dir: Path, poll_interval: float = 0.5, stop: threading.Event | None = None) -> Iterator[dict]`
  Generator that yields parsed record dicts from
  `run_dir/generations/gen_*/rollouts/*.jsonl` in (generation, file order,
  line order). Behavior:
  - On start, yield all existing lines (catch-up), then poll: remember a
    byte offset per file, re-scan the glob each poll so **new generation
    directories and files are picked up**.
  - Tolerate malformed/partial trailing lines: attempt `json.loads`; on
    failure, retry that offset next poll (the writer may be mid-line).
    Never raise out of the generator for bad data.
  - Exit when `stop` is set. Open files read-only.
- `to_event(rec: dict) -> dict` — reduce a record to the wire payload:
  keep `episode_id, generation, step_idx, parsed_action, completion,
  parse_status, reward, cum_reward, terminated, truncated, outcome,
  monitor_flags, model_id, seed`; **drop `prompt_messages` except** add
  `"observed": <content of the last user message, or "">` (what the model
  saw this turn). Use `rec.get(...)` throughout — old-schema files may
  lack fields.

**Verify**: Step 5 unit tests for catch-up, live append, new-directory
pickup, malformed-line tolerance, and payload reduction pass.

### Step 2: The page

`slm_rl/webui/page.py` — module-level `PAGE: str` holding one complete,
self-contained HTML document (inline CSS + vanilla JS, no CDN, no external
requests). Embedding the page as a Python string avoids package-data
packaging entirely. UI requirements:

- Title "SLM-RL — live play". Connects to `/events` via `EventSource`.
- Groups events by `episode_id` into episode cards (newest first, keep the
  latest ~30 cards; drop older from the DOM to bound memory).
- Each step renders: the guess as colored peg circles (map
  `R,G,B,Y,O,P` → red/green/blue/gold/orange/purple; unknown chars render
  as a gray peg with the character), reward, `parse_status` badge
  (non-`ok` highlighted), and any `monitor_flags` keys as small warning
  chips.
- The raw `completion` text renders under the pegs in a `<details>` fold —
  after plan 002 this is where teacher rationales appear.
- Terminal step colors the card border: green for `win`, red for
  `loss`/`truncated`, gray otherwise; show `outcome` and `cum_reward`.
- A header line shows run id, current generation, episodes seen, wins seen.
- Auto-reconnect is native to `EventSource`; also render connection state.

### Step 3: The server

`slm_rl/webui/server.py`:

- `serve(run_dir: Path, host: str = "127.0.0.1", port: int = 8777) -> None`
  using `http.server.ThreadingHTTPServer` + `BaseHTTPRequestHandler`:
  - `GET /` → 200, `text/html`, the `PAGE` string.
  - `GET /events` → 200, `text/event-stream`, then loop over
    `iter_run_records(...)` sending `data: <json(to_event(rec))>\n\n`,
    flushing each event; send a `: keepalive\n\n` comment at least every
    15s while idle. Handle client disconnect (`BrokenPipeError`,
    `ConnectionResetError`) by returning quietly and setting that
    stream's stop event.
  - Anything else → 404. No other routes, no query-driven file paths
    (path traversal surface stays zero: the served run_dir is fixed at
    process start).
- Bind to `127.0.0.1` by default (local viewer, not a public service).

### Step 4: CLI

In `slm_rl/cli.py`, add:

```python
@app.command()
def watch(
    run: str = typer.Option(..., help="Run id under runs/"),
    port: int = typer.Option(8777),
    host: str = typer.Option("127.0.0.1"),
) -> None:
    """Stream a run's episodes live to a browser (read-only observer)."""
    from slm_rl.webui.server import serve  # lazy, like other commands
    ...resolve runs/<run>, error clearly if missing, print the URL, serve...
```

Follow the lazy-import and option style of neighboring commands.

**Verify**: `uv run slm-rl watch --help` shows the command;
`uv run slm-rl watch --run does-not-exist` exits nonzero with a clear message.

### Step 5: Tests

`tests/test_webui.py` (stdlib + pytest only; use `tmp_path`; every server
test binds port 0 and runs in a daemon thread with a hard timeout):

1. **Catch-up**: write 3 record lines across two `gen_*` dirs → generator
   yields 3 dicts in generation order.
2. **Live append + new dir**: start the generator (background thread pushing
   into a queue), append a line to an existing file AND create a new
   `gen_002/rollouts/x.jsonl` with a line → both arrive; then set `stop` and
   join the thread.
3. **Malformed tolerance**: a truncated JSON line yields nothing, and after
   the line is completed on disk it yields exactly once.
4. **Payload**: `to_event` output has no `prompt_messages`, has `observed`
   equal to the last user message content, and survives a record missing
   optional fields.
5. **HTTP smoke**: `serve` on port 0 (grab the bound port), `GET /` contains
   `SLM-RL`; `GET /events` (via `http.client`, read with timeout) receives a
   `data:` line after a record is appended. Shut the server down
   (`server.shutdown()`) in a `finally`.

**Verify**: `uv run pytest -q` → all pass.

### Step 6: Doc note

Add a short subsection to `docs/ARCHITECTURE.md` near the dashboard/
observability material: what `slm-rl watch` is, the read-only invariant, and
that it is stdlib-only by design (8GB rule).

**Verify**: `grep -n "slm-rl watch" docs/ARCHITECTURE.md` → ≥1 match.

## Test plan

Covered in Step 5. The malformed-line and new-directory tests are the ones
that protect against real-world failure (writer races, resumed runs).

## Done criteria

- [ ] `uv run pytest -q` exits 0 (including 5+ new webui tests)
- [ ] `grep -rn "import torch\|import transformers\|import pyarrow\|import streamlit" slm_rl/webui/` → no matches (stdlib only)
- [ ] `grep -n "def watch" slm_rl/cli.py` → 1 match
- [ ] `git diff --stat -- pyproject.toml slm_rl/rollout/ slm_rl/eval/ slm_rl/orchestrator/` → empty
- [ ] Manual smoke against a real run dir (any `runs/<id>` present) renders episode cards
- [ ] `plans/README.md` status row updated

## STOP conditions

- `RolloutRecord` fields differ materially from the Current state excerpt
  (schema drifted; the payload contract needs re-planning).
- You find yourself wanting to modify the rollout writer to make tailing
  easier (e.g. flush control, sidecar files) — that breaks the observer
  invariant; report instead.
- The HTTP smoke test cannot be made reliable without sleeps > 5s (report;
  flaky-by-design tests are worse than fewer tests).

## Maintenance notes

- The wire payload is a *reduction* of `RolloutRecord`; when `schema_version`
  bumps, `to_event` is the single place to reconcile old/new fields.
- Future: an eval-suite viewer would need eval episodes to be recorded
  first (they currently aren't) — that is an orchestrator decision, not a
  webui patch; do not add recording from this package.
- vLLM/batched rollout (plan 005) changes write *interleaving* (multiple
  episodes appending concurrently) but not the schema; the UI already
  groups by `episode_id`, so no change should be needed — verify visually
  after 005 lands.
