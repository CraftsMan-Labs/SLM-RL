#!/usr/bin/env python3
"""Build colab_workshop.ipynb (chapters 0-13).

Re-run from the repo:
    python docs/workshop/build_nb.py

Writes:
    /home/rishub/Desktop/projects/enterprises/craftsmanlabs/SLM-RL/colab_workshop.ipynb

Each chapter is a `chapter_N()` function of `md(...)` / `code(...)` calls,
registered in CHAPTERS. Add or edit a chapter function, then re-run.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(
    "/home/rishub/Desktop/projects/enterprises/craftsmanlabs/SLM-RL/colab_workshop.ipynb"
)

cells: list[dict] = []


def _lines(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").strip("\n") + "\n"
    return text.splitlines(keepends=True)


def md(source: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(source)})


def code(source: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": _lines(source),
        }
    )


MERMAID_JS = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"


def mermaid_cell(graph: str) -> None:
    """Colab renders mermaid only inside %%html, not markdown fences."""
    code(
        "%%html\n"
        f'<script src="{MERMAID_JS}"></script>\n'
        "<script>mermaid.initialize({startOnLoad:true});</script>\n"
        '<div class="mermaid">\n'
        f"{graph.strip()}\n"
        "</div>\n"
    )


# ---------------------------------------------------------------------------
# Chapters. Next agent: copy a chapter_* function, append it to CHAPTERS.
# ---------------------------------------------------------------------------


def chapter_0() -> None:
    md(
        """\
# SLM-RL — Colab workshop

<p>
<img src="https://raw.githubusercontent.com/CraftsMan-Labs/SLM-RL/main/docs/workshop/logo-agentics.svg" height="40" alt="Agentics Foundation"/>
&nbsp;&nbsp;&nbsp;
<img src="https://raw.githubusercontent.com/CraftsMan-Labs/SLM-RL/main/docs/workshop/logo-conscious-engines.svg" height="40" alt="Conscious Engines"/>
</p>

A small language model plays text-Atari, trains on its own games, and keeps the weights **only** if they beat the last champion.

**Runtime → Change runtime type → T4 GPU**, then run top to bottom. Cells call `slm_rl` directly — no web app, frames render inline.
"""
    )
    mermaid_cell(
        """\
flowchart LR
    rollout[ROLLOUT] --> dataset[DATASET]
    dataset --> train[TRAIN]
    train --> ev[EVAL]
    ev --> gate[GATE]
    gate -->|promote| champ[champion]
    gate -->|reject| champ
    champ --> rollout
"""
    )

    md(
        """\
## 0. Setup

Checks the GPU, clones the repo if needed, installs `.[atari]` plus the train stack. Colab already has CUDA torch — do **not** install the `[cuda]` extra.

`PRECISION = "q4"` needs `bitsandbytes`. If that import fails, switch the dropdown to `fp16`.
"""
    )

    code(
        r'''# @title GPU check + clone + install
REPO_URL = "https://github.com/CraftsMan-Labs/SLM-RL.git"  # @param {type:"string"}
BRANCH = "main"  # @param {type:"string"}

import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse


def github_token() -> str | None:
    """Optional Colab Secret / env GITHUB_TOKEN (private forks or rate limits)."""
    try:
        from google.colab import userdata  # type: ignore

        tok = userdata.get("GITHUB_TOKEN")
        if tok:
            return tok.strip()
    except Exception:
        pass
    return (os.environ.get("GITHUB_TOKEN") or "").strip() or None


def authed_clone_url(url: str, token: str | None) -> str:
    if not token:
        return url
    p = urlparse(url)
    if p.scheme != "https" or "github.com" not in (p.netloc or ""):
        return url
    netloc = f"x-access-token:{token}@{p.netloc}"
    return urlunparse((p.scheme, netloc, p.path, "", "", ""))


print("=== GPU ===")
if shutil.which("nvidia-smi"):
    subprocess.check_call(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,compute_cap",
            "--format=csv",
        ]
    )
else:
    print("nvidia-smi not found (CPU runtime?). A T4 is required for the happy path.")


# Prefer an already-uploaded checkout; else clone into /content/SLM-RL.
ROOT = Path("/content/SLM-RL")
here = Path.cwd()
if (here / "pyproject.toml").is_file() and (here / "slm_rl").is_dir():
    ROOT = here
elif (here / "SLM-RL" / "pyproject.toml").is_file():
    ROOT = here / "SLM-RL"
elif not (ROOT / "pyproject.toml").is_file():
    clone_url = authed_clone_url(REPO_URL, github_token())
    print(f"cloning {REPO_URL} @ {BRANCH} → {ROOT}")
    subprocess.check_call(
        ["git", "clone", "--depth", "1", "-b", BRANCH, clone_url, str(ROOT)],
    )
else:
    print(f"already cloned: {ROOT}")

if not (ROOT / "pyproject.toml").is_file():
    raise FileNotFoundError(
        f"Clone failed or incomplete at {ROOT}. "
        f"Check REPO_URL (expected CraftsMan-Labs/SLM-RL) and re-run."
    )

os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])


# Colab already ships CUDA torch — install the package + train libs without
# re-resolving a second torch wheel from the [cuda] extra.
pip("-e", ".[atari]")
pip(
    "transformers>=5.0",
    "trl>=1.8",
    "peft",
    "bitsandbytes",
    "datasets",
    "accelerate",
    "pyarrow",
    "pillow",
    "matplotlib",
    "pandas",
)

import peft
import torch
import transformers
import trl

print("cwd:", Path.cwd())
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print(f"cuda device: {torch.cuda.get_device_name(0)}  capability {major}.{minor}")
else:
    print("cuda device: none — set Runtime → Change runtime type → T4 GPU")

print("--- resolved versions (GRPOConfig names are TRL-version-sensitive) ---")
print(f"torch          {torch.__version__}")
print(f"transformers   {transformers.__version__}")
print(f"trl            {trl.__version__}")
print(f"peft           {peft.__version__}")
try:
    import bitsandbytes

    print(f"bitsandbytes   {bitsandbytes.__version__}")
    print("what just happened: GPU visible, slm_rl installed, bitsandbytes imports.")
except Exception as exc:
    print(f"bitsandbytes import failed: {exc}")
    print(
        'Set PRECISION = "fp16" in the dropdowns two cells down and re-run '
        "that cell. 4-bit loading needs a working bitsandbytes install."
    )
'''
    )

    md(
        """\
### Hardware tiers

`configs/hardware.yaml` is first-match-wins. A T4 (~16 GB) lands on `cuda-8-16gb` → `LFM2.5-1.2B` + GRPO.
"""
    )
    mermaid_cell(
        """\
flowchart TD
    host[detect_host] --> t24{VRAM >= 20 GB?}
    t24 -->|yes| gemma[cuda-24gb / Gemma E2B]
    t24 -->|no| t16{VRAM >= 6 GB?}
    t16 -->|yes| lfm[cuda-8-16gb / LFM 1.2B]
    t16 -->|no| floor[any-8gb / LFM 350M]
"""
    )

    code(
        r'''from slm_rl.platform.hardware import detect_host, resolve_tier
from slm_rl.config.loader import load_tiers

host = detect_host()
tier = resolve_tier(load_tiers())

print(f"OS:          {host.os}")
print(f"RAM:         {host.ram_gb:.1f} GB")
print(f"CUDA VRAM:   {host.cuda_vram_gb if host.cuda_vram_gb is not None else 'none'} GB")
print(f"MPS:         {host.has_mps}")
print("--- resolved tier ---")
print(f"name:        {tier.name}")
print(f"model:       {tier.model}")
print(f"backend:     {tier.backend}")
print(f"train:       {tier.train}")
print(f"quantization:{tier.quantization}")
print(
    "what just happened: detect_host() measured this machine; "
    f"resolve_tier() picked {tier.name!r} from configs/hardware.yaml."
)
'''
    )

    md(
        """\
### Two dropdowns

| Knob | Default | What it does |
|---|---|---|
| `MODE` | `QUICK` | Tiny episode/step counts so each cell finishes in ~1–2 min. `FULL` is a real run. |
| `PRECISION` | `q4` | How weights sit in VRAM. |

- **q4** — 4-bit QLoRA. Lowest VRAM, OOM-proof. Slightly slower on a T4 (dequantize each step). Payoff is headroom: Chapter 10 loads two models.
- **fp16** — 16-bit. Fits a 1.2B LoRA on 16 GB; often a bit faster per step here.
- **auto** — whatever the tier table says.

A T4 has **no bf16** (needs Ampere+). Compute dtype is always fp16 on this GPU.

Colab wipes `/content` on disconnect — uncomment the Drive mount in the next cell if you want runs to survive.
"""
    )

    code(
        r'''# @title Workshop knobs
MODE = "QUICK"       # @param ["QUICK", "FULL"]
PRECISION = "q4"     # @param ["q4", "fp16", "auto"]

from pathlib import Path

import torch

KNOBS = {
    "QUICK": {
        "generations": 2,
        "train": {
            "episodes_per_generation": 4,
            "grpo_max_steps": 12,
            "grpo_max_prompts": 16,
            "group_size": 2,
            "max_completion_tokens": 24,
            "rollout_batch_size": 4,
        },
        "teacher": {"warmstart_episodes": 20},
    },
    "FULL": {
        "generations": 3,
        "train": {
            "episodes_per_generation": 50,
            "grpo_max_steps": 200,
            "grpo_max_prompts": 256,
            "group_size": 8,
            "rollout_batch_size": 8,
        },
    },
}[MODE]
DQN_DECISIONS = {"QUICK": 5_000, "FULL": 300_000}[MODE]
EVAL_LIMIT = {"QUICK": 4, "FULL": 50}[MODE]
BACKEND = {"q4": "transformers-4bit", "fp16": "transformers", "auto": None}[PRECISION]

HOME = "/content/slm-rl-runs"
if not Path("/content").is_dir():
    HOME = str(Path.cwd() / "slm-rl-runs")
Path(HOME).mkdir(parents=True, exist_ok=True)

# Colab wipes /content when the runtime disconnects. To keep runs across sessions:
# from google.colab import drive
# drive.mount("/content/drive")
# HOME = "/content/drive/MyDrive/slm-rl-runs"
# Path(HOME).mkdir(parents=True, exist_ok=True)

resolved_backend = BACKEND or tier.backend
bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
if torch.cuda.is_available():
    free_b, total_b = torch.cuda.mem_get_info()
    vram = f"{free_b / 1024**3:.2f} / {total_b / 1024**3:.2f} GB free/total"
else:
    vram = "no CUDA"

print(f"MODE:              {MODE}")
print(f"PRECISION:         {PRECISION}")
print(f"model:             {tier.model}")
print(f"backend:           {resolved_backend}  (tier default is {tier.backend})")
print(f"bf16 supported:    {bf16}  (T4 = False; compute type is fp16)")
print(f"VRAM:              {vram}")
print(f"HOME:              {HOME}")
print(f"DQN_DECISIONS:     {DQN_DECISIONS}")
print(f"EVAL_LIMIT:        {EVAL_LIMIT}")
print(f"KNOBS.generations: {KNOBS['generations']}")
print(f"KNOBS.train:       {KNOBS['train']}")

if PRECISION == "q4":
    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:
        print(
            f"WARNING: bitsandbytes is not importable ({exc}). "
            "Set PRECISION to fp16 and re-run this cell."
        )

print(
    f"what just happened: workshop knobs resolved for MODE={MODE} "
    f"PRECISION={PRECISION}. Later cells read these names as-is."
)
'''
    )


def viewer_helpers() -> None:
    md(
        """\
### Viewer helpers

Four functions, reused everywhere: `show_frame` (in-place PNG via the repo encoder), `ale_rgb`, `stream_episode` (wraps `game.step` — there is no callback hook), `plot_series`. Diagrams use `%%html` + mermaid.js — Colab markdown fences do not render.
"""
    )

    code(
        r'''# Viewer helpers — defined once, reused by every later chapter.
from IPython.display import Image, display, update_display
from slm_rl.webui.png import encode_rgb

%matplotlib inline

_FRAME_IDS: set[str] = set()


def show_frame(rgb, title, _id):
    """Display (or in-place update) an HxWx3 uint8 frame via the repo PNG encoder."""
    if rgb is None:
        return
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    img = Image(data=encode_rgb(rgb.tobytes(), w, h), format="png", width=min(360, w * 2))
    if _id in _FRAME_IDS:
        update_display(img, display_id=_id)
    else:
        print(title)
        display(img, display_id=_id)
        _FRAME_IDS.add(_id)


def ale_rgb(game):
    """Current ALE screen, or None for non-Atari / unexpected structure.

    GymnasiumGameAdapter stores the env on `_env` (created lazily in reset).
    """
    env = getattr(game, "_env", None) or getattr(game, "env", None)
    try:
        return None if env is None else env.unwrapped.ale.getScreenRGB()
    except Exception:
        return None


def stream_episode(game, every=4):
    """Wrap game.step / game.reset so every Nth decision renders in place."""
    step_fn = getattr(game, "_raw_step", game.step)
    reset_fn = getattr(game, "_raw_reset", game.reset)
    game._raw_step, game._raw_reset = step_fn, reset_fn
    n = {"i": 0}

    def reset(seed=None):
        obs = reset_fn(seed)
        n["i"] = 0
        rgb = ale_rgb(game)
        show_frame(rgb, "live play", "live-frame") if rgb is not None else print(obs.text)
        return obs

    def step(action):
        result = step_fn(action)
        n["i"] += 1
        if n["i"] % every == 0 or result.terminated or result.truncated:
            rgb = ale_rgb(game)
            if rgb is not None:
                show_frame(rgb, "live play", "live-frame")
            else:
                print(result.observation.text)
        return result

    game.reset, game.step = reset, step
    return game


def plot_series(xs, ys, xlabel, ylabel, title):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(list(xs), list(ys))
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(True, alpha=0.3)
    plt.show()


print("what just happened: show_frame, ale_rgb, stream_episode, plot_series are defined.")
'''
    )


def chapter_1() -> None:
    md(
        """\
## 1. The games

Atari as **text**. The model never sees pixels — RAM becomes a short description plus a numbered menu. SLMs are text-native; pixels would need a vision stack.

Types you will keep seeing: `Observation`, `ActionSpec`, `StepResult`. Boxing YAML is 2500 turns; `QUICK` caps `GameConfig.max_turns` to 32.
"""
    )
    mermaid_cell(
        """\
flowchart LR
    ale[ALE RAM] --> render[ram_map]
    render --> obs["Observation.text"]
    obs --> menu["legal_actions"]
    menu --> step["game.step(ActionSpec)"]
"""
    )

    code(
        r'''from slm_rl.agents.bots import RandomAgent
from slm_rl.config.loader import load_game_config
from slm_rl.games.registry import available_games, get_game
from slm_rl.rollout.runner import EpisodeRunner

print("available games:", available_games())

GAME = "boxing"
game_cfg = load_game_config(GAME)
print(f"yaml max_turns: {game_cfg.max_turns}")
if MODE == "QUICK":
    game_cfg = game_cfg.model_copy(update={"max_turns": 32})
    print("QUICK: capped game_cfg.max_turns to 32 so streamed episodes finish.")

game = get_game(GAME)(game_cfg)
obs = game.reset(seed=0)

print("--- system_prompt() ---")
print(game.system_prompt())
print("--- obs.text (what the model reads) ---")
print(obs.text)
print("--- obs.legal_actions (numbered menu) ---")
for i, action in enumerate(obs.legal_actions, start=1):
    print(f"  {i}) id={action.id!r:16s}  label={action.label!r}")

show_frame(ale_rgb(game), "pixels the model never sees — RAM is rendered as the text above", "ch1-reset")

stream_episode(game, every=4)
RANDOM_STATS = EpisodeRunner(
    game, RandomAgent(seed=0), game_cfg,
    run_id="colab", generation=0, model_id="random",
).run_episode(seed=0, episode_id="random-000")

print("--- random episode ---")
print(RANDOM_STATS)
print(
    "what just happened: Boxing was constructed from GameConfig, reset to a "
    "text observation plus a legal-action menu, and a RandomAgent played one "
    f"episode (outcome={RANDOM_STATS.get('outcome')!r})."
)
'''
    )

    md(
        """\
Random play is the baseline. If a method cannot beat uniform legal moves, it has not learned the game.
"""
    )


def chapter_2() -> None:
    md(
        """\
## 2. Config

One merge, low → high. Model id / backend come from the tier table unless you override them (`PRECISION` sets `backend`).

`max_turns` is on `GameConfig`, not `RunConfig` — `game_cfg.model_copy(update={...})`.
"""
    )
    mermaid_cell(
        """\
flowchart LR
    d[default.yaml] --> g[games/boxing.yaml]
    g --> o[overrides / KNOBS]
    o --> cfg[RunConfig]
"""
    )

    code(
        r'''from slm_rl.config.loader import load_run_config

cfg = load_run_config(
    game=GAME,
    overrides={
        "run_id": "colab",
        "home": HOME,
        "backend": BACKEND,
        **KNOBS,
    },
)

print(f"run_id:                       {cfg.run_id}")
print(f"home:                         {cfg.home}")
print(f"game:                         {cfg.game}")
print(f"generations:                  {cfg.generations}")
print(f"backend:                      {cfg.backend}")
print(f"model (None = use the tier):  {cfg.model}")
print(f"train.episodes_per_generation:{cfg.train.episodes_per_generation}")
print(f"train.group_size:             {cfg.train.group_size}")
print(f"train.grpo_max_steps:         {cfg.train.grpo_max_steps}")
print(f"train.grpo_max_prompts:       {cfg.train.grpo_max_prompts}")
print(f"train.rollout_batch_size:     {cfg.train.rollout_batch_size}")
print(f"gate.min_improvement:         {cfg.gate.min_improvement}")
print(f"teacher.warmstart_episodes:   {cfg.teacher.warmstart_episodes}")
print(f"game_cfg.max_turns:           {game_cfg.max_turns}  (GameConfig, not RunConfig)")
print(
    "what just happened: load_run_config merged default.yaml + your overrides. "
    "cfg.model is still None, so later cells use tier.model."
)
'''
    )


def chapter_3() -> None:
    md(
        """\
## 3. The model plays

Aha cell: one Boxing observation → the exact prompt → raw text → parsed `ActionSpec`.

`parse_status`: `ok` / `retry_ok` / `fallback_random` (those last ones count as invalid). First run **downloads ~2 GB**.
"""
    )
    mermaid_cell(
        """\
flowchart LR
    obs[obs.text + menu] --> gen[backend.generate]
    gen --> raw[raw_completion]
    raw --> parse{parse_action}
    parse -->|ACTION / index / fuzzy| ok[ok]
    parse -->|fail| retry[one retry]
    retry -->|fail| rnd[fallback_random]
"""
    )

    code(
        r'''from slm_rl.agents.llm_agent import LLMAgent
from slm_rl.inference.base import GenParams, create_backend

model_id = cfg.model or tier.model
backend_name = BACKEND or tier.backend
print(f"loading {model_id!r} via {backend_name!r} (first time downloads weights)...")
backend = create_backend(backend_name, model_id, tier.quantization)
agent = LLMAgent(
    backend,
    game.system_prompt(),
    GenParams(max_tokens=32, temperature=0.7),
)

obs = game.reset(seed=0)
decision = agent.act(obs)

print("--- prompt messages (exactly what the model saw) ---")
for msg in decision.prompt_messages:
    role, content = msg.get("role"), msg.get("content")
    print(f"[{role}]\n{content}\n")
print("--- raw_completion ---")
print(decision.raw_completion)
print("--- parsed action ---")
print(f"id={decision.action.id!r}  label={decision.action.label!r}")
print(f"parse_status={decision.parse_status!r}")

preview_cfg = game_cfg.model_copy(update={"max_turns": 6})
previous_cfg = game.config
game.config = preview_cfg
stream_episode(game, every=1)
LLM_STATS = EpisodeRunner(
    game, agent, preview_cfg,
    run_id="colab", generation=0, model_id=model_id,
).run_episode(seed=2, episode_id="llm-preview")
game.config = previous_cfg

print("--- a few turns of LLM play ---")
print(LLM_STATS)
print(
    "what just happened: the model received a system prompt + text observation, "
    f"emitted {decision.parse_status!r} text, and played "
    f"{LLM_STATS['steps']} turns (outcome={LLM_STATS.get('outcome')!r})."
)
'''
    )


def chapter_4() -> None:
    md(
        """\
## 4. Rollout + dataset

One JSON line per decision. That file **is** the training data — prompt, completion, action, reward, monitor flags. No need to replay the emulator later.
"""
    )
    mermaid_cell(
        """\
flowchart LR
    agent[LLMAgent] --> runner[EpisodeRunner]
    game[Game] --> runner
    runner --> mon[DoomLoopMonitor]
    mon --> jsonl[RolloutWriter JSONL]
    jsonl --> pq[consolidate parquet]
"""
    )

    code(
        r'''import json
from pathlib import Path

from slm_rl.datagen.writer import RolloutWriter
from slm_rl.rollout.runner import EpisodeRunner

rollout_dir = Path(HOME) / "colab" / "rollouts"
rollout_dir.mkdir(parents=True, exist_ok=True)
jsonl_path = rollout_dir / "gen0.jsonl"

stream_episode(game, every=4)
with RolloutWriter(jsonl_path) as writer:
    runner = EpisodeRunner(
        game, agent, game_cfg, writer=writer,
        run_id="colab", generation=0, model_id=cfg.model or tier.model,
    )
    ROLLOUT_STATS = runner.run_episode(seed=1, episode_id="ep-001")

print("--- EpisodeRunner stats ---")
print(ROLLOUT_STATS)
print("--- one raw JSONL line (schema) ---")
with jsonl_path.open(encoding="utf-8") as fh:
    rec = json.loads(fh.readline())
preview = dict(rec)
preview["prompt_messages"] = f"<{len(rec.get('prompt_messages') or [])} messages>"
print(json.dumps(preview, indent=2, default=str)[:2500])
print(f"jsonl path: {jsonl_path}  ({sum(1 for _ in jsonl_path.open())} lines)")
print(
    "what just happened: one episode was written as JSONL. "
    f"outcome={ROLLOUT_STATS.get('outcome')!r}  steps={ROLLOUT_STATS.get('steps')}  "
    f"invalid_steps={ROLLOUT_STATS.get('invalid_steps')}  "
    f"monitor={ROLLOUT_STATS.get('monitor')}"
)
'''
    )

    md(
        """\
`monitor` is the anti-doom ladder: **reflect → mask_action → truncate**. Boxing YAML enables reflect + truncate only. `BatchedEpisodeRunner` fans one `generate` across K live episodes.

Next cell: `consolidate()` → parquet → `df.head()`.
"""
    )

    code(
        r'''from slm_rl.datagen.consolidate import consolidate
import pandas as pd

parquet_path = Path(HOME) / "colab" / "rollouts.parquet"
n_rows = consolidate(rollout_dir, parquet_path)
df = pd.read_parquet(parquet_path)

print(f"consolidate() wrote {n_rows} rows → {parquet_path}")
print("df.shape:", df.shape)
print("columns:", list(df.columns))
print(df.head())
print(
    "what just happened: every *.jsonl under the rollout directory is now a "
    "parquet table. Nested fields (prompt_messages, monitor_flags) are stored "
    "as JSON strings so the schema stays stable across games."
)
'''
    )


def chapter_5() -> None:
    md(
        """\
## 5. Teachers / hybrid RL

A 1.2B model has never boxed. A DQN has — small, fast, mute. It plays; the SLM studies the traces.

Three seams (`docs/HYBRID_RL.md`): warm-start demos, Q-top-k menu prune, potential shaping. **Hard rule: teachers never touch eval.**
"""
    )
    mermaid_cell(
        """\
flowchart TD
    raw[raw SLM] --> teacher[DQN teacher plays]
    teacher --> hw[homework demos]
    hw --> sft[SFT copies habits]
    sft --> world[SLM plays alone]
    world --> exam[frozen exam — no teacher]
    exam --> topper{beats champion?}
    topper -->|yes| champ[promote]
    topper -->|no| raw
"""
    )

    code(
        r'''import json
from pathlib import Path

import torch

from slm_rl.teachers import make_teacher
from slm_rl.teachers.dqn import metrics_path_for, train_dqn
from slm_rl.teachers.dqn_checkpoint import expected_dqn_checkpoint
from slm_rl.rollout.runner import EpisodeRunner

dqn_device = "cuda" if torch.cuda.is_available() else "cpu"
dqn_path = expected_dqn_checkpoint(GAME, HOME)
print(f"training DQN for {DQN_DECISIONS} decisions on {dqn_device} → {dqn_path}")
DQN_SUMMARY = train_dqn(
    game_cfg,
    decisions=DQN_DECISIONS,
    out_path=dqn_path,
    device=dqn_device,
    seed=0,
)
print("train_dqn summary:", DQN_SUMMARY)

metrics_path = Path(DQN_SUMMARY.get("metrics_path") or metrics_path_for(dqn_path))
xs, ys = [], []
with metrics_path.open(encoding="utf-8") as fh:
    for line in fh:
        row = json.loads(line)
        if row.get("split") == "train" and "decisions" in row and "mean_ep_reward" in row:
            if row["mean_ep_reward"] is None:
                continue
            xs.append(row["decisions"])
            ys.append(row["mean_ep_reward"])
if xs:
    plot_series(xs, ys, "decisions", "mean episode reward (last 20)", "DQN teacher reward curve")
else:
    print("no train-split reward points yet (increase DQN_DECISIONS / switch MODE to FULL).")

teacher_agent, teacher_id = make_teacher(game_cfg, seed=0, dqn_checkpoint=str(dqn_path))
print(f"make_teacher → model_id={teacher_id!r}")

stream_episode(game, every=4)
TEACHER_STATS = EpisodeRunner(
    game, teacher_agent, game_cfg,
    run_id="colab", generation=0, model_id=teacher_id,
).run_episode(seed=3, episode_id="teacher-000")

print("--- scoreboard (same GameConfig, different agents) ---")
print(f"random : {RANDOM_STATS.get('outcome')!r}  cum_reward={RANDOM_STATS.get('cum_reward'):.3f}")
print(f"LLM    : {LLM_STATS.get('outcome')!r}  cum_reward={LLM_STATS.get('cum_reward'):.3f}  (short preview)")
print(f"teacher: {TEACHER_STATS.get('outcome')!r}  cum_reward={TEACHER_STATS.get('cum_reward'):.3f}")
print(
    "what just happened: a CleanRL-pattern DQN trained on RAM vectors, its "
    "reward curve was plotted from the sibling metrics JSONL, and the teacher "
    "played one streamed episode for comparison."
)
'''
    )


def chapter_6() -> None:
    md(
        """\
## 6. Packs

Teacher demos + `dqn.pt` in one folder so a workshop shares homework instead of everyone training the same DQN.

This cell reuses Chapter 5's checkpoint (`dqn_decisions=0`). Push is optional — `HF_TOKEN` in Colab Secrets.
"""
    )
    mermaid_cell(
        """\
flowchart LR
    bake[bake_pack] --> disk["MANIFEST + dqn.pt + rollouts/"]
    disk --> push[push_pack]
    push --> hub[HF dataset]
    hub --> pull[resolve_pack]
    pull --> disk
"""
    )

    code(
        r'''from pathlib import Path

import torch

from slm_rl.hf_auth import apply_hf_token, hf_token
from slm_rl.packs import (
    ATARI_GAMES,
    bake_pack,
    is_atari,
    packs_root,
    push_pack,
    read_manifest,
    resolve_pack,
    write_manifest,
)

print("ATARI_GAMES:", sorted(ATARI_GAMES))
print(f"is_atari({GAME!r}):", is_atari(GAME))
print("packs_root:", packs_root(HOME))

# Pull a Colab Secret into the process env if the attendee set one.
try:
    from google.colab import userdata  # type: ignore

    apply_hf_token(userdata.get("HF_TOKEN"))
except Exception:
    apply_hf_token(hf_token())

BAKE_EPISODES = {"QUICK": 2, "FULL": 20}[MODE]
pack_dir = bake_pack(
    GAME,
    packs_root(HOME),
    episodes=BAKE_EPISODES,
    dqn_decisions=0,  # reuse Chapter 5's checkpoint; do not train another DQN
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=0,
    selection_quantile=1.0 if MODE == "QUICK" else 0.25,
)
print("--- MANIFEST.json ---")
print((pack_dir / "MANIFEST.json").read_text(encoding="utf-8"))
print("read_manifest:", read_manifest(pack_dir))
print(
    "write_manifest is what bake_pack called internally; you can also stamp a "
    "hand-assembled folder with the same function."
)

# Pull a pre-baked pack from Hugging Face by URL (dataset repo). Leave empty to skip.
PACK_URL = ""  # e.g. "https://huggingface.co/datasets/your-org/slm-rl-boxing"
if PACK_URL:
    cached = resolve_pack(PACK_URL, HOME, GAME)
    print("resolve_pack →", cached)
else:
    print(
        "PACK_URL is empty. resolve_pack(url, home, game) downloads a published "
        "dataset pack into packs_root(home) and validates MANIFEST.json."
    )

# Optional publish. Chapter 11 covers this properly; token-gated on purpose.
PUSH_REPO = ""  # e.g. "your-username/slm-rl-boxing"
if PUSH_REPO:
    token = hf_token()
    if not token:
        print("Add HF_TOKEN in Colab Secrets (or export HF_TOKEN) and re-run to push.")
    else:
        commit_url = push_pack(pack_dir, PUSH_REPO, token=token)
        print("push_pack →", commit_url)
else:
    print("PUSH_REPO is empty; local pack stays on disk. write_manifest already ran.")

print(
    f"what just happened: baked a local {GAME} pack at {pack_dir} "
    f"({BAKE_EPISODES} teacher episodes, reused dqn.pt)."
)
'''
    )


def chapter_7() -> None:
    md(
        """\
## 7. Training

Same factory, two strategies. Both write a LoRA adapter (a few MB, not the whole 1.2B).

SFT = learn from your best games. GRPO = sample, score, nudge — slower, needs `GameConfig`. `q4` → QLoRA. T4 compute = fp16. Next cell frees the Chapter 3 backend first.
"""
    )
    mermaid_cell(
        """\
flowchart TB
    subgraph sft [reject_sft — copy the best]
        a[keep top quantile] --> b[prompt / completion pairs]
        b --> c[SFTTrainer]
    end
    subgraph grpo [GRPO — play and score]
        d[one situation] --> e[sample K answers]
        e --> f[score vs the game]
        f --> g[push probability up]
    end
    data[parquet] --> sft
    data --> grpo
    c --> lora[adapter/]
    g --> lora
"""
    )

    code(
        r'''import json
from pathlib import Path

import torch

from slm_rl.datagen.grpo_export import export_grpo_dataset
from slm_rl.datagen.sft_export import export_sft_dataset
from slm_rl.training.lora import release_trainer_memory

# Free the Chapter 3 inference backend before we load a trainable copy.
if "backend" in globals() and backend is not None:
    backend.close()
    backend = None
release_trainer_memory(torch.cuda.is_available())

model_id = cfg.model or tier.model
dataset_path = parquet_path if Path(parquet_path).is_file() else rollout_dir
if not Path(dataset_path).exists():
    raise FileNotFoundError("Chapter 4 dataset is missing — re-run that cell.")

train_dir = Path(HOME) / "colab" / "train"
train_dir.mkdir(parents=True, exist_ok=True)
sft_path = train_dir / "sft.jsonl"
grpo_path = train_dir / "grpo.jsonl"

n_sft = export_sft_dataset(dataset_path, sft_path, cfg.train)
n_grpo = export_grpo_dataset(
    dataset_path, grpo_path, game_cfg, max_prompts=cfg.train.grpo_max_prompts,
)

# Teacher-pack demos are a better SFT source if the LLM episode produced no
# usable pairs (every step was fallback_random).
if n_sft == 0 and "pack_dir" in globals() and (Path(pack_dir) / "rollouts").is_dir():
    print("SFT export from the LLM rollout was empty; falling back to the Chapter 6 pack.")
    n_sft = export_sft_dataset(Path(pack_dir) / "rollouts", sft_path, cfg.train)
    if n_grpo == 0:
        n_grpo = export_grpo_dataset(
            Path(pack_dir) / "rollouts", grpo_path, game_cfg,
            max_prompts=cfg.train.grpo_max_prompts,
        )
    dataset_path = Path(pack_dir) / "rollouts"

def _preview(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        line = fh.readline()
    return json.loads(line) if line.strip() else {}

print(f"SFT rows:  {n_sft}  → {sft_path}")
print("--- one SFT row (prompt / completion pair) ---")
print(json.dumps(_preview(sft_path), indent=2, default=str)[:1800])
print(f"GRPO rows: {n_grpo}  → {grpo_path}")
print("--- one GRPO row (prompt + game_ctx) ---")
print(json.dumps(_preview(grpo_path), indent=2, default=str)[:1800])
print(
    "what just happened: the same decisions were exported twice. SFT is "
    "imitation pairs; GRPO keeps a game_ctx JSON blob the reward functions score."
)
'''
    )

    md(
        """\
`reject_sft` next. Adapter ≈ tens of MB. Base model ≈ a couple of GB. That gap is LoRA.
"""
    )

    code(
        r'''from slm_rl.training.base import create_strategy
from slm_rl.training.lora import release_trainer_memory

four_bit = (BACKEND or tier.backend) == "transformers-4bit"
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

sft_out = Path(HOME) / "colab" / "sft"
sft_strategy = create_strategy(
    "reject_sft", cfg.train, model_id, game_cfg, four_bit=four_bit,
)
SFT_RESULT = sft_strategy.train(dataset_path, sft_out)
print("adapter_path:", SFT_RESULT.adapter_path)
print("metrics:", SFT_RESULT.metrics)
if SFT_RESULT.adapter_path and Path(SFT_RESULT.adapter_path).is_dir():
    print("--- adapter directory ---")
    total = 0
    for p in sorted(Path(SFT_RESULT.adapter_path).rglob("*")):
        if p.is_file():
            total += p.stat().st_size
            print(f"  {p.relative_to(SFT_RESULT.adapter_path)}  {p.stat().st_size / 1024:.1f} KB")
    print(f"total adapter size: {total / 1024**2:.2f} MB  (base model is a couple of GB)")
else:
    print("SFT skipped (no trainable pairs). The gate would see no new adapter.")

if torch.cuda.is_available():
    print(f"peak VRAM after reject_sft: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

del sft_strategy
release_trainer_memory(torch.cuda.is_available())
print(
    "what just happened: reject_sft fitted a LoRA adapter on the best "
    "prompt/completion pairs, then released trainer memory."
)
'''
    )

    md(
        """\
Same factory, `"grpo"`. Free VRAM between the two or the T4 OOMs. Then flip `PRECISION` to `fp16` if you want to compare VRAM vs step time.
"""
    )

    code(
        r'''from slm_rl.training.lora import bf16_ok, compute_dtype, release_trainer_memory

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

grpo_out = Path(HOME) / "colab" / "grpo"
grpo_strategy = create_strategy(
    "grpo", cfg.train, model_id, game_cfg, four_bit=four_bit,
)
GRPO_RESULT = grpo_strategy.train(dataset_path, grpo_out)
print("adapter_path:", GRPO_RESULT.adapter_path)
print("metrics:", GRPO_RESULT.metrics)
if GRPO_RESULT.adapter_path and Path(GRPO_RESULT.adapter_path).is_dir():
    total = 0
    for p in sorted(Path(GRPO_RESULT.adapter_path).rglob("*")):
        if p.is_file():
            total += p.stat().st_size
    print(f"GRPO adapter size: {total / 1024**2:.2f} MB")

print("--- resolved precision ---")
print(f"bf16_ok():              {bf16_ok()}  (T4 must be False)")
print(f"compute_dtype(True):    {compute_dtype(True)}")
print(f"four_bit:               {four_bit}  (from backend {(BACKEND or tier.backend)!r})")
if torch.cuda.is_available():
    print(f"peak VRAM after GRPO:   {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"VRAM free/total now:    {free_b / 1024**3:.2f} / {total_b / 1024**3:.2f} GB")
else:
    print("no CUDA — peak VRAM is n/a")

del grpo_strategy
release_trainer_memory(torch.cuda.is_available())

ADAPTER_PATH = None
for result in (GRPO_RESULT, SFT_RESULT):
    p = getattr(result, "adapter_path", None)
    if p and Path(p).is_dir():
        ADAPTER_PATH = Path(p)
        break
print(f"ADAPTER_PATH for later cells: {ADAPTER_PATH}")
print(
    "what just happened: GRPO sampled completions, scored them, and wrote "
    "another adapter. Trainer memory is released again."
)
'''
    )


def chapter_8() -> None:
    md(
        """\
## 8. Eval and the gate

Same frozen seeds, no teacher, no pruner. Promote only if the SLM itself got better.

Guards: `min_improvement`, `max_invalid_rate`, `max_intervention_rate_ratio`, `min_mean_entropy`. Boxing primary = `mean_score`. The cell also sabotages a copy so you see a real "no".
"""
    )
    mermaid_cell(
        """\
flowchart TD
    suite[eval_suite seeds] --> base[base model]
    suite --> cand[adapter]
    base --> g{EvalGate.decide}
    cand --> g
    g -->|margin + hygiene| yes[promote]
    g -->|else| no[reject]
"""
    )

    code(
        r'''from slm_rl.agents.llm_agent import LLMAgent
from slm_rl.eval.gate import EvalGate
from slm_rl.eval.suites import run_suite
from slm_rl.games.registry import get_game
from slm_rl.inference.base import GenParams, create_backend
from slm_rl.training.lora import release_trainer_memory

game_cls = get_game(GAME)
suite = game_cls.eval_suite()
print(f"suite: game={suite.game} primary={suite.primary_metric} n_seeds={len(suite.seeds)} limit={EVAL_LIMIT}")

eval_backend = create_backend(BACKEND or tier.backend, model_id, tier.quantization)
eval_params = GenParams(max_tokens=cfg.train.max_completion_tokens, temperature=0.2)

def _make_eval_agent():
    return LLMAgent(eval_backend, game.system_prompt(), eval_params)

print("evaluating base model...")
BASE_METRICS = run_suite(suite, _make_eval_agent, game_cls, game_cfg, limit=EVAL_LIMIT)
print("base:", BASE_METRICS)

if ADAPTER_PATH is not None:
    eval_backend.load_adapter(ADAPTER_PATH)
    print(f"evaluating adapter at {ADAPTER_PATH}...")
    CANDIDATE_METRICS = run_suite(suite, _make_eval_agent, game_cls, game_cfg, limit=EVAL_LIMIT)
else:
    print("no adapter on disk — candidate metrics copy the base (training was skipped).")
    CANDIDATE_METRICS = dict(BASE_METRICS)

eval_backend.close()
release_trainer_memory(torch.cuda.is_available())

print("--- side by side ---")
keys = ("primary", "mean_score", "win_rate", "invalid_rate", "intervention_rate", "episodes")
print(f"{'metric':22s}  {'base':>10s}  {'candidate':>10s}")
for key in keys:
    b, c = BASE_METRICS.get(key), CANDIDATE_METRICS.get(key)
    def _fmt(v):
        return f"{v:10.4f}" if isinstance(v, float) else f"{v!s:>10s}"
    print(f"{key:22s}  {_fmt(b)}  {_fmt(c)}")

gate = EvalGate(cfg.gate)
promote, reason = gate.decide(BASE_METRICS, CANDIDATE_METRICS)
print(f"gate (real candidate): promote={promote}  reason={reason}")

worse = dict(CANDIDATE_METRICS)
worse["primary"] = float(CANDIDATE_METRICS.get("primary") or 0.0) - 1.0
worse["invalid_rate"] = max(float(CANDIDATE_METRICS.get("invalid_rate") or 0.0), cfg.gate.max_invalid_rate + 0.1)
fake_promote, fake_reason = gate.decide(BASE_METRICS, worse)
print(f"gate (deliberately worse): promote={fake_promote}  reason={fake_reason}")
print(
    "what just happened: the frozen suite was played twice, the gate judged "
    "the real adapter, then rejected a sabotaged copy so you can see it say no."
)
'''
    )


def chapter_9() -> None:
    md(
        """\
## 9. The evolve loop

One generation = one pass. Promotion moves the champion pointer; reject leaves it put.

`GenerationRunner` reloads game YAML (Boxing = 2500 turns / 100 evals). The cell writes a `config_dir` overlay so QUICK stays short. `ensure_baseline()` is gen 0, cached.

QUICK may **not** improve. That's the gate working, not a bug.
"""
    )
    mermaid_cell(
        """\
flowchart LR
    R[ROLLOUT] --> D[DATASET]
    D --> T[TRAIN]
    T --> E[EVAL]
    E --> G[GATE]
    G -->|promote| C[champion]
    G -->|reject| C
    C --> R
"""
    )

    code(
        r'''import json
from pathlib import Path

import yaml

from slm_rl.orchestrator.generation import GenerationRunner
from slm_rl.orchestrator.paths import RunPaths
from slm_rl.training.lora import release_trainer_memory

release_trainer_memory(torch.cuda.is_available())

WORKSHOP_CONFIG_DIR = Path(HOME) / "colab" / "configs"
(WORKSHOP_CONFIG_DIR / "games").mkdir(parents=True, exist_ok=True)
workshop_game = game_cfg.model_copy(update={"eval_episodes": EVAL_LIMIT})
(WORKSHOP_CONFIG_DIR / "games" / f"{GAME}.yaml").write_text(
    yaml.safe_dump(workshop_game.model_dump()), encoding="utf-8",
)

evolve_runner = GenerationRunner(cfg, config_dir=WORKSHOP_CONFIG_DIR)
print(f"model={evolve_runner.model_id} backend={evolve_runner.backend_name} strategy={evolve_runner.strategy_name}")
print(f"game_cfg.max_turns={evolve_runner.game_cfg.max_turns} eval_episodes={evolve_runner.game_cfg.eval_episodes}")
print(f"registry next_generation={evolve_runner.registry.next_generation} champion={evolve_runner.registry.champion}")

baseline = evolve_runner.ensure_baseline()
print("--- gen 0 baseline ---")
print(baseline)

start = evolve_runner.registry.next_generation
stop = start + cfg.generations
EVOLVE_METRICS = []
for g in range(start, stop):
    m = evolve_runner.run_generation(g)
    EVOLVE_METRICS.append((g, m))
    gate = m.get("gate") or {}
    rollout = m.get("rollout") or {}
    ev = m.get("eval") or {}
    print(f"=== generation {g} ===")
    print(f"  rollout: episodes={rollout.get('episodes')}  train_win_rate={rollout.get('train_win_rate')}")
    print(f"  train:   {m.get('train')}")
    print(f"  eval:    primary={ev.get('primary')}  invalid_rate={ev.get('invalid_rate')}")
    print(f"  gate:    promoted={gate.get('promoted')}  reason={gate.get('reason')}")

run_paths = RunPaths(cfg.home, cfg.run_id)
print("--- registry.json ---")
print(run_paths.registry.read_text(encoding="utf-8"))

xs, ys = [], []
g0 = run_paths.generation(0) / "eval" / "results.json"
if g0.is_file():
    xs.append(0)
    ys.append(float(json.loads(g0.read_text())["primary"]))
for metrics_file in sorted(run_paths.root.glob("generations/gen_*/metrics.json")):
    gen = int(metrics_file.parent.name.split("_")[1])
    primary = json.loads(metrics_file.read_text()).get("eval", {}).get("primary")
    if primary is not None:
        xs.append(gen)
        ys.append(float(primary))
        print(f"gen_{gen:03d}/metrics.json primary={primary}")
if xs:
    plot_series(xs, ys, "generation", evolve_runner.suite.primary_metric, "evolve: primary score by generation")

print(
    "what just happened: GenerationRunner ran baseline + "
    f"{len(EVOLVE_METRICS)} generation(s). Promotion moves the champion pointer; "
    "rejection leaves it put."
)
'''
    )


def chapter_10() -> None:
    md(
        """\
## 10. Theater

Payoff: base vs champion on the **same** seeds. Exhibition, not eval — eval is never written to disk.

`run_exhibition` loads one model at a time. We replay JSONL — no second load.
"""
    )
    mermaid_cell(
        """\
flowchart LR
    seeds["seeds >= 20_000"] --> base[base plays]
    seeds --> champ[champion plays]
    base --> jsonl[theater JSONL]
    champ --> jsonl
    jsonl --> replay[replay actions + hstack frames]
"""
    )

    code(
        r'''import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image as PILImage, ImageDraw

from slm_rl.games.base import ActionSpec
from slm_rl.games.registry import get_game
from slm_rl.orchestrator.paths import RunPaths
from slm_rl.theater.exhibition import run_exhibition
from slm_rl.training.lora import release_trainer_memory

release_trainer_memory(torch.cuda.is_available())

run_dir = RunPaths(cfg.home, cfg.run_id).root
THEATER_EPISODES = {"QUICK": 1, "FULL": 3}[MODE]
print(f"run_dir={run_dir}  episodes={THEATER_EPISODES}")

EXHIBITION = run_exhibition(
    run_dir, GAME,
    episodes=THEATER_EPISODES,
    seed_start=20_000,
    config_dir=WORKSHOP_CONFIG_DIR,
)
print(f"base_dir:             {EXHIBITION.base_dir}")
print(f"champion_dir:         {EXHIBITION.champion_dir}")
print(f"champion_generation:  {EXHIBITION.champion_generation}")
print(f"message:              {EXHIBITION.message}")


def _jsonl_episodes(side_dir: Path | None) -> dict[str, list[dict]]:
    if side_dir is None:
        return {}
    files = sorted(Path(side_dir).glob("generations/gen_*/rollouts/*.jsonl"))
    episodes: dict[str, list[dict]] = defaultdict(list)
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                episodes[rec["episode_id"]].append(rec)
    for steps in episodes.values():
        steps.sort(key=lambda r: r["step_idx"])
    return dict(episodes)


def _mean_score(episodes: dict[str, list[dict]]) -> float:
    scores = []
    for steps in episodes.values():
        last = steps[-1]
        out = last.get("outcome") or ""
        if isinstance(out, str) and out.startswith("score:"):
            scores.append(float(out.split(":", 1)[1]))
        else:
            scores.append(float(last.get("cum_reward") or 0.0))
    return sum(scores) / len(scores) if scores else float("nan")


def _compose(left, right, left_s, right_s, step: int):
    if left is None and right is None:
        return None
    if left is None:
        left = np.zeros_like(right)
    if right is None:
        right = np.zeros_like(left)
    pair = np.concatenate([left, right], axis=1)
    bar_h = 28
    canvas = PILImage.new("RGB", (pair.shape[1], pair.shape[0] + bar_h), (20, 20, 20))
    canvas.paste(PILImage.fromarray(pair), (0, bar_h))
    ImageDraw.Draw(canvas).text(
        (8, 6),
        f"step {step}   BASE {left_s}   CHAMP {right_s}",
        fill=(240, 240, 240),
    )
    return np.asarray(canvas)


base_eps = _jsonl_episodes(EXHIBITION.base_dir)
champ_eps = _jsonl_episodes(EXHIBITION.champion_dir)
print(f"mean score  base={_mean_score(base_eps):.3f}  champion={_mean_score(champ_eps):.3f}")

base_steps = next(iter(base_eps.values())) if base_eps else []
champ_steps = next(iter(champ_eps.values())) if champ_eps else []
n = max(len(base_steps), len(champ_steps))
if n == 0:
    print("no theater JSONL to replay.")
else:
    cls = get_game(GAME)
    g_base = cls(workshop_game)
    g_champ = cls(workshop_game)
    seed_b = int(base_steps[0]["seed"]) if base_steps else 20_000
    seed_c = int(champ_steps[0]["seed"]) if champ_steps else seed_b
    obs_b = g_base.reset(seed=seed_b)
    obs_c = g_champ.reset(seed=seed_c)
    show_frame(
        _compose(ale_rgb(g_base), ale_rgb(g_champ), 0, 0, 0),
        "theater: base (left) vs champion (right)",
        "theater-ab",
    )
    for i in range(n):
        if i < len(base_steps):
            aid = base_steps[i]["parsed_action"]
            spec = next((a for a in obs_b.legal_actions if a.id == aid), ActionSpec(aid, aid))
            res = g_base.step(spec)
            obs_b = res.observation
            score_b = obs_b.metadata.get("score", res.info.get("outcome"))
        else:
            score_b = obs_b.metadata.get("score")
        if i < len(champ_steps):
            aid = champ_steps[i]["parsed_action"]
            spec = next((a for a in obs_c.legal_actions if a.id == aid), ActionSpec(aid, aid))
            res = g_champ.step(spec)
            obs_c = res.observation
            score_c = obs_c.metadata.get("score", res.info.get("outcome"))
        else:
            score_c = obs_c.metadata.get("score")
        if i % 4 == 0 or i + 1 == n:
            show_frame(
                _compose(ale_rgb(g_base), ale_rgb(g_champ), score_b, score_c, i + 1),
                "theater: base (left) vs champion (right)",
                "theater-ab",
            )

print(
    "what just happened: run_exhibition wrote paired JSONL; we replayed the "
    "recorded actions in two fresh games and hstacked the ALE screens."
)
'''
    )


def chapter_11() -> None:
    md(
        """\
## 11. Publish

Colab Secrets → `HF_TOKEN` (write scope). Missing token = friendly no-op, never a crash.
"""
    )
    mermaid_cell(
        """\
flowchart LR
    run[run_dir] --> pub[publish_experiment]
    pub --> model["username/slm-rl-colab"]
    pub --> data["username/slm-rl-colab-data"]
"""
    )

    code(
        r'''from pathlib import Path

from slm_rl.datagen.hf_publish import publish_experiment
from slm_rl.hf_auth import apply_hf_token, hf_token
from slm_rl.orchestrator.paths import RunPaths

token = None
try:
    from google.colab import userdata  # type: ignore

    token = userdata.get("HF_TOKEN")
except Exception:
    token = None
token = apply_hf_token(token)

if not token:
    print(
        "No HF_TOKEN found. Add one in Colab Secrets (key icon, left sidebar) "
        "with write scope, or export HF_TOKEN, then re-run this cell. "
        "Nothing was uploaded."
    )
    PUBLISH_RESULT = None
else:
    from huggingface_hub import HfApi

    who = HfApi(token=token).whoami()
    username = who.get("name")
    run_dir = RunPaths(cfg.home, cfg.run_id).root
    print(f"publishing experiment {cfg.run_id!r} as {username}/slm-rl-{cfg.run_id} (token present, not printed)")
    PUBLISH_RESULT = publish_experiment(
        token=token,
        username=username,
        experiment=cfg.run_id,
        game=GAME,
        run_dir=run_dir,
    )
    print(PUBLISH_RESULT.to_json())
    if PUBLISH_RESULT.dataset_repo:
        print("dataset:", f"https://huggingface.co/datasets/{PUBLISH_RESULT.dataset_repo}")
    if PUBLISH_RESULT.model_repo:
        print("model:  ", f"https://huggingface.co/{PUBLISH_RESULT.model_repo}")

print(
    "what just happened: "
    + (
        "publish_experiment uploaded (or reported a partial failure on) the run."
        if PUBLISH_RESULT is not None
        else "publish was skipped because no write token is configured."
    )
)
'''
    )


def chapter_12() -> None:
    md(
        """\
## 12. Build your own game

Pure Python, seed-deterministic, no ML imports. Required: `reset`, `step`, `state_hash`, `system_prompt`, `eval_suite`.

The cell registers `guess-number` and rolls it out with the same runner as Boxing.
"""
    )
    mermaid_cell(
        """\
flowchart LR
    abc[Game ABC] --> reg["@register_game"]
    reg --> runner[EpisodeRunner]
    runner --> rest[export / train / eval / gate]
"""
    )

    code(
        r'''import hashlib
import random
from pathlib import Path

from slm_rl.agents.bots import RandomAgent
from slm_rl.config.schema import GameConfig
from slm_rl.datagen.writer import RolloutWriter
from slm_rl.eval.suites import EvalSuite
from slm_rl.games.base import ActionSpec, Game, Observation, StepResult
from slm_rl.games.registry import available_games, get_game, register_game
from slm_rl.rollout.runner import EpisodeRunner


@register_game("guess-number")
class GuessNumberGame(Game):
    def reset(self, seed=None):
        rng = random.Random(seed)
        self._secret = rng.randint(1, 7)
        self._turn = 0
        self._low, self._high = 1, 7
        return self._obs("I picked a number. Guess it.")

    def _menu(self):
        return [ActionSpec(id=str(n), label=f"guess {n}") for n in range(self._low, self._high + 1)]

    def _obs(self, text):
        return Observation(text=text, legal_actions=self._menu(), turn=self._turn)

    def step(self, action):
        guess = int(action.id)
        self._turn += 1
        if guess == self._secret:
            return StepResult(self._obs("Correct."), 1.0, True, False, {"outcome": "win"})
        if guess < self._secret:
            self._low = max(self._low, guess + 1)
            hint = "too low"
        else:
            self._high = min(self._high, guess - 1)
            hint = "too high"
        truncated = self._turn >= self.config.max_turns
        text = f"{guess} is {hint}. Range is now {self._low}-{self._high}."
        info = {"outcome": "loss"} if truncated else {}
        return StepResult(self._obs(text), -0.1, False, truncated, info)

    def state_hash(self):
        raw = f"{self._secret}:{self._turn}:{self._low}:{self._high}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def system_prompt(self):
        return "Guess the secret integer. Reply with one line: ACTION: <number>."

    @classmethod
    def eval_suite(cls):
        return EvalSuite(game="guess-number", seeds=tuple(range(100, 110)), primary_metric="win_rate")


print("available games now:", available_games())
guess_cfg = GameConfig(name="guess-number", max_turns=6)
guess_game = get_game("guess-number")(guess_cfg)
guess_path = Path(HOME) / "colab" / "guess-number.jsonl"
with RolloutWriter(guess_path) as writer:
    GUESS_STATS = EpisodeRunner(
        guess_game, RandomAgent(seed=0), guess_cfg, writer=writer,
        run_id="colab", generation=0, model_id="random",
    ).run_episode(seed=0, episode_id="guess-000")
print(GUESS_STATS)
print(f"jsonl lines: {sum(1 for _ in guess_path.open())}  path={guess_path}")
print(
    "what just happened: a brand-new game was registered in-process and "
    "rolled out with the same EpisodeRunner as Boxing. The pipeline did not change."
)
'''
    )

    md(
        """\
Nothing else changed — everything speaks the `Game` ABC.

```toml
[project.entry-points."slm_rl.games"]
guess-number = "my_pkg.guess:GuessNumberGame"
```
"""
    )


def chapter_13() -> None:
    md(
        """\
## 13. Tests

Fast slice: Boxing, config merge, JSONL writer. Not the full suite.
"""
    )

    code(
        r'''!python -m pytest -q -x tests/test_boxing.py tests/test_config.py tests/test_rollout_writer.py
print("what just happened: a fast subset of the repo tests ran and stopped on the first failure if any.")
'''
    )

    md(
        """\
## Appendix

Skipped on purpose: Vue playground / Docker, MLX, vLLM, vision backend, hosted ELO.

Docs: `ARCHITECTURE` · `PIPELINE` · `HYBRID_RL` · `PLUGIN_GUIDE` · `LIFECYCLE`

| Symptom | Fix |
|---|---|
| CUDA OOM | `PRECISION=q4`, `MODE=QUICK`, `backend.close()` |
| bitsandbytes fails | `PRECISION=fp16` |
| Session died | Re-run Ch. 0 + knobs, or uncomment Drive mount |
| Slow first load | ~2 GB download; cached after that |
"""
    )


CHAPTERS = [
    chapter_0,
    viewer_helpers,
    chapter_1,
    chapter_2,
    chapter_3,
    chapter_4,
    chapter_5,
    chapter_6,
    chapter_7,
    chapter_8,
    chapter_9,
    chapter_10,
    chapter_11,
    chapter_12,
    chapter_13,
]


def main() -> None:
    for build in CHAPTERS:
        build()
    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    OUT.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
