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

import html
import json
import sys
from pathlib import Path

WORKSHOP = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSHOP))

from talk_track import chapter_by_number, colab_cue  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "colab_workshop.ipynb"
ASSET_RAW = (
    "https://raw.githubusercontent.com/CraftsMan-Labs/SLM-RL/main/"
    "docs/workshop/assets"
)
DIAGRAM_RAW = f"{ASSET_RAW}/diagrams"
DECK_RAW = f"{ASSET_RAW}/deck"
MARIO_RAW = f"{ASSET_RAW}/mario"

cells: list[dict] = []
_CURRENT_CHAPTER: int | None = None


def _lines(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").strip("\n") + "\n"
    return text.splitlines(keepends=True)


def md(source: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(source)})


def code(source: str) -> None:
    text = source.replace("\r\n", "\n").strip("\n") + "\n"
    lines = text.splitlines(keepends=True)
    # Keep the workshop focused on controls and outputs. Colab still exposes
    # every source block through "Show code" when an attendee wants details.
    metadata: dict = {"cellView": "form"}
    if lines and lines[0].startswith("# @title") and "{display-mode:" not in lines[0]:
        lines[0] = lines[0].rstrip("\n") + ' {display-mode: "form"}\n'
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": metadata,
            "outputs": [],
            "source": lines,
        }
    )


# Native deck stills are ~1920px; unconstrained markdown `![]()` overflows a
# 13-inch Colab pane. Keep stills compact so cells stay on one laptop screen.
_MEDIA_STYLE = (
    "max-width:min(100%,480px);max-height:min(28vh,240px);"
    "width:auto;height:auto;object-fit:contain;display:block;"
)
_THUMB_STYLE = (
    "max-width:min(100%,180px);max-height:100px;"
    "width:auto;height:auto;object-fit:contain;display:block;"
)


def _img_tag(src: str, caption: str, style: str = _MEDIA_STYLE) -> str:
    alt = html.escape(caption, quote=True)
    return f'<img src="{src}" alt="{alt}" style="{style}">'


def _capped_img(src: str, caption: str) -> None:
    md(f"{_img_tag(src, caption)}\n\n*{caption}*")


def _capped_video(src: str, caption: str) -> None:
    md(
        f'<video src="{src}" controls muted playsinline '
        f'style="{_MEDIA_STYLE}"></video>\n\n*{caption}*'
    )


def diagram_cell(name: str, caption: str) -> None:
    """Static SVG. Colab shows the image, not diagram source."""
    _capped_img(f"{DIAGRAM_RAW}/{name}.svg", caption)


def deck_image(name: str, caption: str) -> None:
    """Still from the Vue deck (`docs/workshop/assets/deck`)."""
    _capped_img(f"{DECK_RAW}/{name}", caption)


def deck_video(name: str, caption: str) -> None:
    """Short clip from the Vue deck. Colab markdown plays HTML5 video."""
    _capped_video(f"{DECK_RAW}/{name}", caption)


def mario_video(name: str, caption: str) -> None:
    """World 1-1 clip recorded from checksummed public CNN checkpoints."""
    _capped_video(f"{MARIO_RAW}/{name}", caption)


def chapter_open(number: int, body: str) -> None:
    global _CURRENT_CHAPTER
    _CURRENT_CHAPTER = number
    row = chapter_by_number(number)
    md(
        f"{row['heading']}\n\n"
        f"**Goal.** {row['goal']}\n\n"
        f"**{colab_cue(number)}**\n\n"
        f"{body.rstrip()}\n"
    )
    # Chapter 0 starts after the participant key is bound in "Join the room".
    if number > 0:
        code(
            f"# Track progress — chapter {number} started\n"
            "from lab import start_chapter\n"
            f"start_chapter({number})\n"
        )


def challenge(title: str, body: str) -> None:
    """Always-visible prompt. The next form cell is the attendee's answer."""
    md(f"> **Your move — {title}**\n>\n> {body}")


def optional_challenge(title: str, body: str) -> None:
    """Skip-safe extra. The main path does not depend on this cell."""
    md(f"<details>\n<summary>Optional challenge — {title}</summary>\n\n{body}\n\n</details>")


def checkpoint(phase: str, artifacts: str, nxt: str, *, chapter: int | None = None) -> None:
    md(
        f"### Checkpoint — {phase}\n\n"
        f"| | |\n|---|---|\n"
        f"| Produced | {artifacts} |\n"
        f"| Next | {nxt} |\n"
    )
    n = _CURRENT_CHAPTER if chapter is None else chapter
    if n is not None:
        code(
            f"# Track progress — chapter {n} complete\n"
            "from lab import complete_chapter\n"
            f"complete_chapter({n})\n"
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
<img src="https://raw.githubusercontent.com/CraftsMan-Labs/SLM-RL/main/docs/workshop/logo-ideas2it.svg" height="40" alt="Ideas2IT"/>
</p>

A small language model plays text-Atari, trains on its own games, and keeps the weights **only** if they beat the last champion.

**Runtime → Change runtime type → T4 GPU**, then run cells in order. A few cells ask you to **type** — that is on purpose. `Runtime → Run all` pauses there so the room stays together.

Each chapter: **Choose → Predict → Run → Observe**. Code starts collapsed so the controls and results stay readable; click **Show code** whenever you want the implementation. Yellow form cells (`# @param`) keep their knobs visible. Typed questions are required brakes: blank answers do not continue. Challenge cells are extra — skip those and the rest still runs.

The deck on the other screen keeps the story. This notebook is the execution surface. Chapter headings name the matching slides.
"""
    )
    deck_image("HeroVisual.png", "Observe → act → reward → improve. Small models. Big moves.")
    diagram_cell("evolve-loop", "ROLLOUT → DATASET → TRAIN → EVAL → GATE → champion")
    deck_video("FreeWay_trained.mp4", "Today we make a model play — trained Freeway from the deck.")
    chapter_open(
        0,
        "Checks the GPU, clones the repo if needed, installs `.[atari]` plus the train stack. "
        "Colab already has CUDA torch — do **not** install the `[cuda]` extra.\n\n"
        '`PRECISION = "q4"` needs `bitsandbytes`. If that import fails, switch the dropdown to `fp16`.',
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

# Colab keeps /content across notebook re-runs. Refresh a clean checkout so an
# old Mario action map cannot silently turn checkpoint action 0 into NOOP.
if str(ROOT).startswith("/content/") and (ROOT / ".git").is_dir():
    dirty = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(
            f"{ROOT} has local changes, so it was not auto-updated. "
            "Restart the runtime for a clean workshop checkout."
        )
    print(f"syncing {ROOT} to origin/{BRANCH}")
    subprocess.check_call(
        ["git", "-C", str(ROOT), "fetch", "--depth", "1", "origin", BRANCH],
    )
    subprocess.check_call(
        ["git", "-C", str(ROOT), "checkout", "-B", BRANCH, "FETCH_HEAD"],
    )

os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for stale_module in ("mario_lab", "playable"):
    sys.modules.pop(stale_module, None)


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
    "ipywidgets",
    "ipyevents",
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
    diagram_cell("hardware-tier", "First-match hardware tiers from configs/hardware.yaml")

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
### Workshop knobs

Yellow form at the top of the next cell. Change a value, re-run **this cell**, then continue — later chapters read these names.

| Knob | Default | What it does |
|---|---|---|
| `MODE` | `QUICK` | Tiny episode/step counts so each cell finishes in ~1–2 min. `FULL` is a real run (20–40+ min later). |
| `PRECISION` | `q4` | How weights sit in VRAM. |
| `GAME` | `boxing` | Workshop Atari title. Re-run from here through Chapter 1 if you switch. |
| `SEED` | `0` | Shared RNG so two attendees with the same seed can compare. |
| `RUN_NAME` | `colab` | Folder under `HOME`. Letters, digits, `-`, `_`. |

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
GAME = "boxing"      # @param ["boxing", "space-invaders", "freeway", "demon-attack"]
SEED = 0             # @param {type:"integer"}
RUN_NAME = "colab"   # @param {type:"string"}

import sys
from pathlib import Path

import torch

_WS = Path.cwd() / "docs" / "workshop"
if str(_WS) not in sys.path:
    sys.path.insert(0, str(_WS))
from lab import (  # noqa: E402
    bound,
    clamp_int,
    resolve_backend,
    resolve_game,
    resolve_mode,
    sanitize_run_name,
    scorecard,
)

MODE = resolve_mode(MODE)
GAME = resolve_game(GAME)
PRECISION = PRECISION if PRECISION in ("q4", "fp16", "auto") else "q4"
SEED = clamp_int(SEED, *bound(MODE, "seed"), "SEED")
RUN_NAME = sanitize_run_name(RUN_NAME)

KNOBS = {
    "QUICK": {
        "generations": 2,
        "train": {
            # Cap of QUICK bounds: more SFT homework without flipping to FULL.
            "episodes_per_generation": 8,
            "grpo_max_steps": 12,
            "grpo_max_prompts": 16,
            "group_size": 2,
            "max_completion_tokens": 24,
            "rollout_batch_size": 4,
        },
        # Stronger gen-1 distill; keep reject_sft wall-clock in workshop range.
        "teacher": {"warmstart_episodes": 40},
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
EVAL_LIMIT = {"QUICK": 8, "FULL": 50}[MODE]
BACKEND = resolve_backend(PRECISION)

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
    gpu_name = torch.cuda.get_device_name(0)
else:
    vram = "no CUDA"
    gpu_name = "none"

if MODE == "FULL":
    print("FULL is a real run — later cells can take 20–40+ minutes. QUICK is the workshop default.")
if torch.cuda.is_available() and "T4" not in gpu_name:
    print(f"NOTE: GPU is {gpu_name}, not a T4. q4 is still the safe default.")

if PRECISION == "q4":
    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:
        print(
            f"WARNING: bitsandbytes is not importable ({exc}). "
            "Set PRECISION to fp16 and re-run this cell."
        )

scorecard(
    "resolved knobs",
    [
        ("MODE", MODE),
        ("PRECISION", PRECISION),
        ("GAME", GAME),
        ("SEED", SEED),
        ("RUN_NAME", RUN_NAME),
        ("model", tier.model),
        ("backend", f"{resolved_backend}  (tier default {tier.backend})"),
        ("bf16 supported", f"{bf16}  (T4 = False; compute type is fp16)"),
        ("GPU", gpu_name),
        ("VRAM", vram),
        ("HOME", HOME),
        ("DQN_DECISIONS", DQN_DECISIONS),
        ("EVAL_LIMIT", EVAL_LIMIT),
        ("generations", KNOBS["generations"]),
        ("train", KNOBS["train"]),
    ],
)
print("If you change GAME, re-run from this cell through Chapter 1.")
print(
    f"what just happened: workshop knobs resolved for MODE={MODE} "
    f"PRECISION={PRECISION} GAME={GAME}."
)
'''
    )

    md(
        """\
### Join the room

Progress is tracked live on [WorkShopTracker](https://workshop.craftsmanlabs.net/signin).

1. Open this run’s join link:
   **[https://workshop.craftsmanlabs.net/join/slm-rl-test-run-056892](https://workshop.craftsmanlabs.net/join/slm-rl-test-run-056892)**
2. Sign in, join the run, then **Generate key** on `/workshop`.
3. Run the next yellow cell and paste the secret into **`WST_API_KEY`** (or store it as Colab Secret `WST_API_KEY`). The value is **never printed**.

That key is **run-scoped and yours** — never paste an admin/master key into this notebook.

`Runtime → Run all` still stops until you type a display name. That name is what the local scorecard uses at the end.

Paste a Hugging Face token in the yellow form, or wait for the hidden prompt. Create a **write** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). The token is applied to the process env and **never printed**. Blank is allowed — Hub downloads stay anonymous, and Chapter 11 publish becomes a no-op.
"""
    )

    code(
        r'''# @title Join the room
DISPLAY_NAME = "anonymous"  # @param {type:"string"}
WORKSHOP_JOIN_URL = "https://workshop.craftsmanlabs.net/join/slm-rl-test-run-056892"  # @param {type:"string"}
HF_TOKEN = ""  # @param {type:"string"}

from lab import ask, ask_hf_token, new_card, show_card
from slm_rl.hf_auth import apply_hf_token, hf_token

name = ask(
    "Your name (shown on the scorecard)",
    default=DISPLAY_NAME,
)
DISPLAY_NAME = name
CARD = new_card(name)

join_url = (WORKSHOP_JOIN_URL or "").strip()
print(f"Join / generate key: {join_url}")
print("Next cell: paste your participant API key (shown once on /workshop).")

seeded = (HF_TOKEN or "").strip() or (hf_token() or "")
if not seeded:
    try:
        from google.colab import userdata  # type: ignore

        seeded = (userdata.get("HF_TOKEN") or "").strip()
    except Exception:
        seeded = ""
HF_TOKEN = apply_hf_token(ask_hf_token(default=seeded)) or ""
if HF_TOKEN:
    print("HF token applied (not printed). Hub downloads and optional publish will use it.")
else:
    print(
        "No HF token — Hub stays anonymous. Paste one in this form and re-run "
        "if a gated model download fails, or before Chapter 11 publish."
    )

show_card(CARD)
print(
    f"welcome, {CARD['name']}. later cells will pause and ask you to type. "
    "that is the brake on Runtime → Run all."
)
'''
    )

    md(
        """\
### Load your WorkShopTracker API key

Paste the **participant** secret from `/workshop` into the yellow field (or leave blank if Colab Secret `WST_API_KEY` is already set). Then run the cell — it connects to the live dashboard and starts Chapter 0.
"""
    )

    code(
        r'''# @title Load WorkShopTracker API key
WST_API_KEY = ""  # @param {type:"string"}

import os

from lab import ask_wst_api_key, connect_workshop_tracker, start_chapter

join_url = globals().get("WORKSHOP_JOIN_URL") or (
    "https://workshop.craftsmanlabs.net/join/slm-rl-test-run-056892"
)
join_url = str(join_url).strip()

# Form → Colab Secret → env → hidden prompt. Value is never printed.
key = ask_wst_api_key(default=WST_API_KEY)
if key:
    os.environ["WST_API_KEY"] = key
    WST_API_KEY = ""  # clear the form variable so re-runs don't keep echoing it in UI state

TRACKER = connect_workshop_tracker(join_url=join_url, api_key=key or None)
if TRACKER is not None:
    start_chapter(0)
else:
    print("Tracker not connected — progress cells will skip until you re-run with a key.")
'''
    )


def viewer_helpers() -> None:
    md(
        """\
### Viewer helpers

Four viewers plus a few workshop utilities: `show_frame`, `ale_rgb`, `stream_episode`, `plot_series`, `unwrap_game`, `ensure_game`, `close_backend_if_any`. Pipeline diagrams are static SVGs so Colab never shows diagram source.
"""
    )

    code(
        r'''# Viewer + workshop helpers — defined once, reused by every later chapter.
import sys
from pathlib import Path

from IPython.display import Image, display, update_display
from slm_rl.webui.png import encode_rgb

%matplotlib inline

_WS = Path.cwd() / "docs" / "workshop"
if str(_WS) not in sys.path:
    sys.path.insert(0, str(_WS))
from lab import require_names  # noqa: E402

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


def unwrap_game(game):
    """Restore original step/reset if stream_episode wrapped them."""
    if hasattr(game, "_raw_step"):
        game.step = game._raw_step
        game.reset = game._raw_reset
        del game._raw_step, game._raw_reset
    return game


def stream_episode(game, every=4):
    """Wrap game.step / game.reset so every Nth decision renders in place."""
    unwrap_game(game)
    step_fn = game.step
    reset_fn = game.reset
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


def ensure_game():
    """Rebuild game/game_cfg when GAME or MODE changed, and unwrap stale stream wrappers."""
    global game, game_cfg
    from slm_rl.config.loader import load_game_config
    from slm_rl.games.registry import get_game

    require_names(globals(), "GAME", "MODE")
    key = (GAME, MODE)
    current = globals().get("game")
    if current is not None and getattr(current, "_workshop_key", None) == key:
        return unwrap_game(current), game_cfg
    game_cfg = load_game_config(GAME)
    if MODE == "QUICK":
        game_cfg = game_cfg.model_copy(update={"max_turns": 32})
    game = get_game(GAME)(game_cfg)
    game._workshop_key = key
    return game, game_cfg


def close_backend_if_any(name="backend"):
    """Close a cached inference backend and free trainer CUDA cache."""
    import torch
    from slm_rl.training.lora import release_trainer_memory

    be = globals().get(name)
    if be is not None:
        try:
            be.close()
        except Exception:
            pass
        globals()[name] = None
    release_trainer_memory(torch.cuda.is_available())


print("what just happened: viewers, unwrap_game, ensure_game, close_backend_if_any are defined.")

from lab import complete_chapter

complete_chapter(0)
'''
    )


def chapter_1() -> None:
    chapter_open(
        1,
        "Atari as **text**. The model never sees pixels — RAM becomes a short description plus a numbered menu. "
        "SLMs are text-native; pixels would need a vision stack.\n\n"
        "Play the title yourself first — clickable buttons, not a keyboard. Mario installs only if you pick it.\n\n"
        "Types you will keep seeing: `Observation`, `ActionSpec`, `StepResult`. "
        "Boxing YAML is 2500 turns; `QUICK` caps `GameConfig.max_turns` to 32.",
    )
    deck_image("SLM.png", "An SLM is a coachable specialist — smaller, cheaper, faster to experiment.")
    deck_image("Why-Games.png", "Games are measurable sandboxes: clear actions, cheap failures, repeatable eval.")
    diagram_cell("games-pipeline", "ALE RAM → text observation → legal menu → game.step")

    code(
        r'''from slm_rl.agents.bots import RandomAgent
from slm_rl.games.registry import available_games
from slm_rl.rollout.runner import EpisodeRunner
from lab import require_names, scorecard

require_names(globals(), "GAME", "MODE", "SEED", "RUN_NAME")
print("available games:", available_games())

game, game_cfg = ensure_game()
print(f"effective max_turns: {game_cfg.max_turns}")
if MODE == "QUICK":
    print("QUICK: capped game_cfg.max_turns to 32 so streamed episodes finish.")

obs = game.reset(seed=SEED)
print("--- system_prompt() ---")
print(game.system_prompt())
print("--- obs.text (what the model reads) ---")
print(obs.text)
print("--- obs.legal_actions (numbered menu) ---")
for i, action in enumerate(obs.legal_actions, start=1):
    print(f"  {i}) id={action.id!r:16s}  label={action.label!r}")

show_frame(ale_rgb(game), "pixels the model never sees — RAM is rendered as the text above", "ch1-reset")
print("Play the game with clickable buttons next, then pick one typed menu line.")
'''
    )

    md(
        """\
### Play before you train

Click inside the game once, then use **Arrow keys** or **WASD**. Press **Space/X** to punch, fire, or jump. The controls below the screen are the fallback and expose every legal action. Watch the frame, reward, and score move — this is the same game a DQN will later train on.

Pick `mario` only when you want the NES emulator. That install is lazy and must not break Atari if it fails.
"""
    )

    code(
        r'''# @title Play before you train
PLAY_GAME = "boxing"   # @param ["boxing", "space-invaders", "freeway", "demon-attack", "mario"]
INSTALL_MARIO = True   # @param {type:"boolean"}

from IPython.display import SVG, display
from lab import require_names
from playable import GamePanel, make_playable, show_game_panel

require_names(globals(), "GAME")
if "PLAY_PANEL" in globals() and PLAY_PANEL is not None:
    try:
        PLAY_PANEL.close()
    except Exception:
        pass

PLAY_ENV, PLAY_ERROR = make_playable(
    PLAY_GAME,
    install_mario=bool(INSTALL_MARIO) and PLAY_GAME == "mario",
)
if PLAY_ENV is None:
    print(f"playable {PLAY_GAME!r} unavailable: {PLAY_ERROR}")
    PLAY_PANEL = None
    if PLAY_GAME == "mario":
        from mario_lab import fallback_paths

        story = fallback_paths()["storyboard"]
        if story.is_file():
            display(SVG(filename=str(story)))
        print("Atari is still available — set PLAY_GAME back to boxing / freeway / …")
    else:
        print("Re-run Chapter 1's first cell if the Atari env was not built yet.")
else:
    PLAY_PANEL = GamePanel(PLAY_ENV)
    show_game_panel(PLAY_PANEL)
    print(
        f"what just happened: keyboard + clickable controls for {PLAY_GAME}. "
        "Click the game, use Arrow keys/WASD, and press Space/X for the action. "
        "Reset starts over; repeat is capped so a click cannot run away."
    )
'''
    )

    md(
        """\
You have now played the same environment the agents use. A random-policy baseline is generated automatically alongside the DQN teacher in Chapter 5.
"""
    )
    checkpoint("games", "`game` / `game_cfg` plus your playable panel", "merge config in Chapter 2")


def chapter_2() -> None:
    chapter_open(
        2,
        "One merge, low → high. Model id / backend come from the tier table unless you override them "
        "(`PRECISION` sets `backend`).\n\n"
        "`max_turns` is on `GameConfig`, not `RunConfig` — `game_cfg.model_copy(update={...})`.",
    )
    diagram_cell("config-merge", "default.yaml → game YAML → form overrides → RunConfig")

    challenge(
        "who wins the merge?",
        "If `default.yaml`, the game YAML, and the form overrides disagree, which value is in `RunConfig`?",
    )

    code(
        r'''# @title Config overrides
MERGE_WINNER = "overrides"     # @param ["default.yaml", "game yaml", "overrides"]
EPISODES_OVERRIDE = 0          # @param {type:"integer"}

from slm_rl.config.loader import load_run_config
from lab import bound, clamp_int, ensure_card, grade, record_guess, require_names, scorecard

require_names(globals(), "GAME", "HOME", "BACKEND", "KNOBS", "MODE", "RUN_NAME", "SEED")
CARD = ensure_card(globals())
game, game_cfg = ensure_game()
episodes = int(KNOBS["train"]["episodes_per_generation"])
if EPISODES_OVERRIDE:
    lo, hi = bound(MODE, "episodes_per_generation")
    episodes = clamp_int(EPISODES_OVERRIDE, int(lo), int(hi), "EPISODES_OVERRIDE")

overrides = {
    "run_id": RUN_NAME,
    "home": HOME,
    "backend": BACKEND,
    "seed": SEED,
    **KNOBS,
}
overrides.setdefault("train", {})
overrides["train"] = {**KNOBS.get("train", {}), "episodes_per_generation": episodes}

cfg = load_run_config(game=GAME, overrides=overrides)

scorecard(
    "RunConfig",
    [
        ("run_id", cfg.run_id),
        ("home", cfg.home),
        ("game", cfg.game),
        ("seed", cfg.seed),
        ("generations", cfg.generations),
        ("backend", cfg.backend),
        ("model (None = tier)", cfg.model),
        ("episodes_per_generation", cfg.train.episodes_per_generation),
        ("group_size", cfg.train.group_size),
        ("grpo_max_steps", cfg.train.grpo_max_steps),
        ("grpo_max_prompts", cfg.train.grpo_max_prompts),
        ("rollout_batch_size", cfg.train.rollout_batch_size),
        ("gate.min_improvement", cfg.gate.min_improvement),
        ("teacher.warmstart_episodes", cfg.teacher.warmstart_episodes),
        ("game_cfg.max_turns", f"{game_cfg.max_turns}  (GameConfig, not RunConfig)"),
        ("merge quiz", grade(MERGE_WINNER, "overrides")),
    ],
)
record_guess(CARD, "merge winner", MERGE_WINNER, "overrides")
print(
    "what just happened: load_run_config merged default.yaml + game YAML + your overrides. "
    "Last writer wins — that is the form. cfg.model is still None, so later cells use tier.model."
)
'''
    )
    checkpoint("config", "`cfg` (RunConfig) + resolved episodes", "watch the model take one action")


def chapter_3() -> None:
    chapter_open(
        3,
        "Aha cell: one observation → the exact prompt → raw text → parsed `ActionSpec`.\n\n"
        "`parse_status`: `ok` / `retry_ok` / `fallback_random` (those last ones count as invalid). "
        "First run **downloads ~2 GB**.",
    )
    diagram_cell("parse-action", "Generate a completion, parse an action, or fall back to random")

    challenge(
        "will it parse?",
        "Type your guess **before** the model speaks — `Runtime → Run all` waits. "
        "A raw `1.2B` often emits `ACTION: FIRE` (`ok`) — or garbage (`fallback_random`). "
        "Temperature 0.0 is more obedient; 1.2 is noisier.",
    )

    code(
        r'''# @title Model play knobs
TEMPERATURE = 0.7              # @param {type:"slider", min:0.0, max:1.5, step:0.1}
MAX_TOKENS = 32                # @param {type:"slider", min:8, max:64, step:8}
PREDICT_PARSE = "ok"           # @param ["ok", "retry_ok", "fallback_random"]

from slm_rl.agents.llm_agent import LLMAgent
from slm_rl.inference.base import GenParams, create_backend
from lab import ask, bound, clamp_float, clamp_int, ensure_card, grade, record_guess, require_names, scorecard

require_names(globals(), "cfg", "tier", "BACKEND", "MODE", "SEED", "RUN_NAME")
CARD = ensure_card(globals())
PREDICT_PARSE = ask(
    "Guess the parse status before the model speaks",
    allowed=("ok", "retry_ok", "fallback_random"),
    default=PREDICT_PARSE,
)
game, game_cfg = ensure_game()
lo_t, hi_t = bound(MODE, "temperature")
lo_k, hi_k = bound(MODE, "max_tokens")
TEMPERATURE = clamp_float(TEMPERATURE, lo_t, hi_t, "TEMPERATURE")
MAX_TOKENS = clamp_int(MAX_TOKENS, int(lo_k), int(hi_k), "MAX_TOKENS")

close_backend_if_any("backend")
model_id = cfg.model or tier.model
backend_name = BACKEND or tier.backend
print(f"loading {model_id!r} via {backend_name!r} (first time downloads weights)...")
backend = create_backend(backend_name, model_id, tier.quantization)
agent = LLMAgent(
    backend,
    game.system_prompt(),
    GenParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE),
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
print(grade(PREDICT_PARSE, str(decision.parse_status)))
record_guess(CARD, "parse status", PREDICT_PARSE, str(decision.parse_status))

preview_cfg = game_cfg.model_copy(update={"max_turns": 6})
previous_cfg = game.config
game.config = preview_cfg
unwrap_game(game)
stream_episode(game, every=1)
LLM_STATS = EpisodeRunner(
    game, agent, preview_cfg,
    run_id=RUN_NAME, generation=0, model_id=model_id,
).run_episode(seed=SEED + 2, episode_id="llm-preview")
game.config = previous_cfg
unwrap_game(game)

scorecard(
    "LLM preview",
    [
        ("temperature", TEMPERATURE),
        ("max_tokens", MAX_TOKENS),
        ("parse_status", decision.parse_status),
        ("action", f"{decision.action.id}  ({decision.action.label})"),
        ("steps", LLM_STATS.get("steps")),
        ("outcome", LLM_STATS.get("outcome")),
        ("cum_reward", LLM_STATS.get("cum_reward")),
        ("invalid_steps", LLM_STATS.get("invalid_steps")),
    ],
)
print(
    "what just happened: the model received a system prompt + text observation, "
    f"emitted {decision.parse_status!r} text, and played "
    f"{LLM_STATS['steps']} turns (outcome={LLM_STATS.get('outcome')!r})."
)
'''
    )
    checkpoint("model play", "`backend`, `agent`, `LLM_STATS`", "write one JSONL episode")


def chapter_4() -> None:
    chapter_open(
        4,
        "One JSON line per decision. That file **is** the training data — prompt, completion, action, "
        "reward, monitor flags. No need to replay the emulator later.",
    )
    deck_image("SpaceInv.png", "Take an action, then score what happened. You control the scoreboard.")
    deck_image(
        "Observer-Learn-Play-React.png",
        "SFT: guiding a hand. RL: play alone and score the outcome.",
    )
    diagram_cell("rollout-dataset", "EpisodeRunner writes JSONL, then consolidate to parquet")

    code(
        r'''import json
from pathlib import Path

from slm_rl.datagen.writer import RolloutWriter
from slm_rl.rollout.runner import EpisodeRunner
from lab import require_names, scorecard

require_names(globals(), "HOME", "RUN_NAME", "SEED", "cfg", "agent", "tier")
game, game_cfg = ensure_game()

rollout_dir = Path(HOME) / RUN_NAME / "rollouts"
rollout_dir.mkdir(parents=True, exist_ok=True)
jsonl_path = rollout_dir / "gen0.jsonl"

unwrap_game(game)
stream_episode(game, every=4)
with RolloutWriter(jsonl_path) as writer:
    runner = EpisodeRunner(
        game, agent, game_cfg, writer=writer,
        run_id=RUN_NAME, generation=0, model_id=cfg.model or tier.model,
    )
    ROLLOUT_STATS = runner.run_episode(seed=SEED + 1, episode_id="ep-001")
unwrap_game(game)

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

Next cell: `consolidate()` → parquet → pick a row.
"""
    )

    challenge(
        "what is the training target?",
        "Which field is the completion the trainer copies? "
        "`parsed_action` is the environment action id; `completion` is the "
        "model text persisted in the rollout dataset.",
    )

    code(
        r'''# @title Inspect a rollout row
ROW_INDEX = 0                         # @param {type:"integer"}
SCHEMA_FIELD = "completion"           # @param ["prompt_messages", "completion", "parsed_action", "reward", "monitor_flags"]

from slm_rl.datagen.consolidate import consolidate
import pandas as pd
from lab import bound, clamp_int, grade, require_names, scorecard

require_names(globals(), "HOME", "RUN_NAME", "rollout_dir", "MODE")
parquet_path = Path(HOME) / RUN_NAME / "rollouts.parquet"
n_rows = consolidate(rollout_dir, parquet_path)
df = pd.read_parquet(parquet_path)
lo, hi = bound(MODE, "row_index")
row_i = clamp_int(ROW_INDEX, 0, max(0, len(df) - 1), "ROW_INDEX")
row = df.iloc[row_i].to_dict() if len(df) else {}

scorecard(
    f"row {row_i} / {len(df)}",
    [
        ("shape", df.shape),
        ("columns", list(df.columns)),
        ("parsed_action", row.get("parsed_action")),
        ("reward", row.get("reward")),
        ("completion", str(row.get("completion"))[:180]),
        ("schema quiz", grade(SCHEMA_FIELD, "completion")),
    ],
)
print(df.head())
print(
    "what just happened: every *.jsonl under the rollout directory is now a "
    "parquet table. Nested fields (prompt_messages, monitor_flags) are stored "
    "as JSON strings so the schema stays stable across games."
)
'''
    )
    checkpoint("dataset", "`jsonl_path`, `parquet_path`, `df`", "train a mute DQN teacher")


def chapter_5() -> None:
    chapter_open(
        5,
        "A 1.2B model has never played this title. A DQN has — small, fast, mute. It plays; the SLM studies the traces.\n\n"
        "Three seams (`docs/HYBRID_RL.md`): warm-start demos, Q-top-k menu prune, potential shaping. "
        "**Hard rule: teachers never touch eval.**\n\n"
        "Same pattern as the deck: idea → Mario → the name. "
        "Three short checkpoints explain Q-values, Bellman targets, and ε. "
        "Then train Mario live and let a trained policy play for a user-set step budget. "
        "The Atari teacher cell stays the critical path; live Mario falls back to recorded clips.",
    )
    diagram_cell("dqn-hybrid", "DQN teacher writes homework; the SLM is examined without the teacher")
    deck_video("meet-the-teacher.mp4", "Meet the teacher — a mute DQN already knows useful moves.")
    md(
        """\
**The job.** A mute teacher writes homework. A 1.2B model has never played this title; a small DQN has. The SLM will study the traces. The teacher never sits the exam.
"""
    )

    md(
        """\
**Why not a table?** A Q-table needs one row per screen. Mario has too many. A neural net estimates a number per action instead — those numbers are Q-values. The loop: look, score every move, pick. Sometimes try something new.
"""
    )
    diagram_cell(
        "dqn-loop",
        "State → online net → Q-values → ε-greedy choice → chosen action (current best guess) → environment → reward + next state.",
    )

    md(
        """\
**One Mario frame.** Two buttons: `RIGHT` vs `RIGHT+A`. The net scored this frame:

| Action | Q-value |
|---|---|
| `RIGHT` | 0.41 |
| `RIGHT+A` | 1.27 |

Biggest number is the current best guess — not a proven optimal policy.
"""
    )
    deck_image(
        "dqn-game-master.png",
        "Game master: compare every legal move, then point to the strongest future.",
    )
    diagram_cell("dqn-q-values", "One state, a number per action. The biggest number is the move.")
    challenge(
        "which move should Mario choose?",
        "Treat each Q-value like the DQN’s predicted future score for that move. "
        "Higher is better: compare `RIGHT = 0.41` with `RIGHT+A = 1.27`, "
        "then choose the move you expect the DQN to take.",
    )
    code(
        r'''# @title Which move should Mario choose?
# @markdown A **Q-value** is the DQN's estimate of how useful a move will be. The move with the **higher number wins**.
# @markdown **RIGHT** = run forward without jumping. **RIGHT+A** = run forward while holding Mario's **A (jump)** button.
# @markdown `RIGHT` = **0.41** &nbsp;&nbsp; | &nbsp;&nbsp; `RIGHT+A` = **1.27**
PREDICT_ACTION = "RIGHT"   # @param ["RIGHT", "RIGHT+A"]

from lab import ask, ensure_card, grade, record_guess, scorecard

CARD = ensure_card(globals())
SAMPLE_Q = {"RIGHT": 0.41, "RIGHT+A": 1.27}
CHOSEN = max(SAMPLE_Q, key=SAMPLE_Q.get)
PREDICT_ACTION = ask(
    "Which move wins because it has the higher Q-value?",
    allowed=("RIGHT", "RIGHT+A"),
    default=PREDICT_ACTION,
)
print("Mario's choices:")
print("  RIGHT    0.41  ███")
print("  RIGHT+A  1.27  ██████████  ← higher")
print(f"\nDQN chooses {CHOSEN}: 1.27 is greater than 0.41.")
print(grade(PREDICT_ACTION, CHOSEN))
record_guess(CARD, "RIGHT vs RIGHT+A", PREDICT_ACTION, CHOSEN)
scorecard(
    "Q-values",
    [
        ("RIGHT", SAMPLE_Q["RIGHT"]),
        ("RIGHT+A", SAMPLE_Q["RIGHT+A"]),
        ("your guess", PREDICT_ACTION),
        ("choice", CHOSEN),
    ],
)
'''
    )

    md(
        """\
**Today plus leftover.** A jump is the coin you grab now plus almost all of the best move after you land. That sum is the Bellman target. Predicted Q chases it. The loss is just “prediction versus target.” No derivation.

`target = reward + γ × best next Q` for a continuing run. At death or episode end there is no “tomorrow,” so `target = reward`.
"""
    )
    deck_image(
        "dqn-today-plus-tomorrow.png",
        "A move is today’s reward plus the position it leaves behind.",
    )
    diagram_cell("dqn-bellman", "Reward now + discounted best next action (γ = 0.99).")
    diagram_cell("dqn-math", "target = reward + γ × best next Q. Predicted value chases that target.")
    challenge(
        "compute the target",
        "The DQN predicted a value for Mario's last move. After seeing what "
        "actually happened, it creates a better answer: reward now plus the "
        "discounted value of the best move available next. This better answer "
        "is the Bellman target.",
    )
    code(
        r'''# @title One Bellman target
# @markdown The **Bellman target** is the improved answer used to teach the DQN what its previous move was worth.
# @markdown **Target = reward received now + γ × value of the best next move**
# @markdown `REWARD` = points received after the move. `GAMMA` = how much future value matters (`0.99` keeps 99%).
# @markdown `NEXT_Q_*` = predicted values of moves on the next screen. `PREDICTED_Q` = the DQN's old estimate for the previous move.
# @markdown If `TERMINAL` is checked, the episode ended, so there is no next move and no future value.
REWARD = 1.0                 # @param {type:"number"}
GAMMA = 0.99                 # @param {type:"number"}
NEXT_Q_RIGHT = 0.50          # @param {type:"number"}
NEXT_Q_RIGHT_A = 2.10        # @param {type:"number"}
PREDICTED_Q = 1.20           # @param {type:"number"}
TERMINAL = False             # @param {type:"boolean"}

from lab import clamp_float, scorecard

reward = clamp_float(REWARD, -2.0, 5.0, "REWARD")
gamma = clamp_float(GAMMA, 0.0, 1.0, "GAMMA")
q_right = clamp_float(NEXT_Q_RIGHT, -5.0, 10.0, "NEXT_Q_RIGHT")
q_right_a = clamp_float(NEXT_Q_RIGHT_A, -5.0, 10.0, "NEXT_Q_RIGHT_A")
predicted = clamp_float(PREDICTED_Q, -5.0, 10.0, "PREDICTED_Q")
best_next = max(q_right, q_right_a)
best_name = "RIGHT+A" if q_right_a >= q_right else "RIGHT"
future = 0.0 if TERMINAL else gamma * best_next
target = reward + future
gap = target - predicted
direction = "increase" if gap > 0 else ("decrease" if gap < 0 else "keep")

print("Step 1 — reward received now:")
print(f"  {reward:.3f}")
if TERMINAL:
    print("Step 2 — episode ended, so future value is 0.")
else:
    print("Step 2 — choose the best estimated move on the next screen:")
    print(f"  {best_name} has the highest next Q-value: {best_next:.3f}")
    print("Step 3 — discount that future value:")
    print(f"  {gamma:.2f} × {best_next:.3f} = {future:.3f}")
print("Step 4 — build the teaching target:")
print(f"  {reward:.3f} + {future:.3f} = {target:.3f}")
print("Step 5 — compare with the DQN's old prediction:")
print(f"  old prediction {predicted:.3f}; target {target:.3f}")
print(f"  learning should {direction} the prediction by {abs(gap):.3f}.")

scorecard(
    "Bellman",
    [
        ("target", f"{target:.3f}  =  reward {reward} + future {future:.3f}"),
        ("terminal?", TERMINAL),
        ("best next action", f"{best_name}  Q={best_next}"),
        ("current prediction", predicted),
        ("target − prediction", round(gap, 3)),
        ("learning direction", direction),
    ],
)
print(
    "what just happened: you computed one Bellman target and compared it "
    "with the net's current guess. The update pulls predicted Q toward the target."
)
'''
    )

    md(
        """\
**Shuffled flashcards.** If you only restudy the last page, you memorize one streak. Write every moment on a card — `(state, action, reward, next)` — and mix the pile. That pile is a **replay buffer**. A batch is a random draw, not the last frames.
"""
    )
    deck_image(
        "dqn-flashcards-target.png",
        "Replay shuffled experience against a temporarily frozen answer key.",
    )
    diagram_cell("dqn-replay", "Play once, study many times. Random batches break the streak.")

    md(
        """\
**A frozen answer key.** If the key changes while you mark the test, nobody learns. The **online** net plays. The **target** net is a delayed copy that writes `reward + γ × best next Q`. Copy it forward only now and then.
"""
    )
    diagram_cell("dqn-target", "Online net plays. Target net grades. Then they sync.")

    md(
        """\
**Explore early, trust later.** Wander widely at first. Later follow the best-known route — and still turn down a new alley about one time in twenty. The die-roll is **ε-greedy**. The greedy pick is the current best guess, not a truly optimal policy.
"""
    )
    deck_image(
        "dqn-tourist-then-regular.png",
        "ε-greedy: trust the learned route, but occasionally explore a side path.",
    )
    challenge(
        "set ε",
        "Pick an ε. The cell draws a small seeded sample: exploit = current best guess "
        "(`RIGHT+A`); explore = a random legal button.",
    )
    code(
        r'''# @title Epsilon sample
# @markdown **ε (epsilon)** is the chance of exploring instead of choosing the current best-known action. `0` = always exploit; `1` = always explore.
EPSILON = 0.20               # @param {type:"slider", min:0.0, max:1.0, step:0.05}
N_DRAWS = 20                 # @param {type:"integer"}

import random

from lab import clamp_float, clamp_int, require_names, scorecard

require_names(globals(), "SEED")
eps = clamp_float(EPSILON, 0.0, 1.0, "EPSILON")
n = clamp_int(N_DRAWS, 5, 40, "N_DRAWS")
legal = ("RIGHT", "RIGHT+A")
exploit_action = "RIGHT+A"
rng = random.Random(SEED)
rows = []
for i in range(n):
    if rng.random() < eps:
        choice = rng.choice(legal)
        tag = "explore"
    else:
        choice = exploit_action
        tag = "exploit"
    rows.append((tag, choice))
n_explore = sum(tag == "explore" for tag, _ in rows)
scorecard(
    "ε-greedy sample",
    [
        ("ε", eps),
        ("draws", n),
        ("seed", SEED),
        ("exploit (current best guess)", exploit_action),
        ("explore / exploit", f"{n_explore} / {n - n_explore}"),
        ("sequence", " ".join(f"{tag[0]}:{choice}" for tag, choice in rows)),
    ],
)
print(
    "what just happened: a deterministic ε-greedy sample. "
    "High ε wanders; low ε trusts the current best guess."
)
'''
    )

    md(
        """\
**Watch learning happen.** The next cell does real DQN updates for a workshop-length budget. Default is a public warm-start so improvement is visible. From-scratch is allowed and will look weak.

Loss is “prediction versus target,” not how far Mario ran.
"""
    )

    code(
        r'''# @title Train Mario live
TRAINING_MODE = "warm-start"   # @param ["warm-start", "from-scratch"]
TRAIN_MINUTES = 15             # @param {type:"slider", min:1, max:20, step:1}
EVAL_INTERVAL = 400            # @param {type:"integer"}
SAVE_TO_DRIVE = False          # @param {type:"boolean"}
MARIO_MODEL_REPO = "BLANK/mario-dqn-workshop"  # @param {type:"string"}
MARIO_MODEL_REVISION = "main"  # @param {type:"string"}

from pathlib import Path

from IPython.display import SVG, display
from lab import clamp_float, clamp_int, require_names, scorecard
from mario_lab import (
    TRAIN_MINUTES_RANGE,
    fallback_paths,
    load_fallback_metrics,
    train_mario_live,
)

require_names(globals(), "HOME", "SEED")
minutes = clamp_float(TRAIN_MINUTES, TRAIN_MINUTES_RANGE[0], TRAIN_MINUTES_RANGE[1], "TRAIN_MINUTES")
interval = clamp_int(EVAL_INTERVAL, 50, 2000, "EVAL_INTERVAL")
MARIO_HOME = Path(HOME) / "mario-demo"
if TRAINING_MODE == "from-scratch":
    print("from-scratch: a short run may not produce competent play.")

def _on_chunk(row):
    ev = row.get("eval") or {}
    print(
        f"chunk decisions={row.get('decisions')}  ε={row.get('epsilon')}  "
        f"loss={row.get('loss')}  distance={ev.get('max_x_pos')}  "
        f"reward={ev.get('sum_reward')}"
    )

MARIO_TRAIN = train_mario_live(
    MARIO_HOME,
    training_mode=TRAINING_MODE,
    train_minutes=minutes,
    eval_interval=interval,
    seed=SEED,
    save_to_drive=SAVE_TO_DRIVE,
    repo_id=MARIO_MODEL_REPO,
    revision=MARIO_MODEL_REVISION,
    on_progress=_on_chunk,
)
scorecard(
    "Mario live train",
    [
        ("mode", MARIO_TRAIN.get("mode")),
        ("training", MARIO_TRAIN.get("training_mode")),
        ("source", MARIO_TRAIN.get("source") or MARIO_TRAIN.get("reason")),
        ("device", MARIO_TRAIN.get("device")),
        ("minutes", MARIO_TRAIN.get("train_minutes")),
        ("decisions", MARIO_TRAIN.get("decisions")),
        ("deaths", MARIO_TRAIN.get("deaths")),
        ("checkpoint", MARIO_TRAIN.get("checkpoint")),
        ("chunks", len(MARIO_TRAIN.get("history") or [])),
    ],
)
hist = MARIO_TRAIN.get("history") or []
if hist:
    plot_series(
        [r["decisions"] for r in hist],
        [r["eval"]["max_x_pos"] for r in hist],
        "decisions",
        "eval distance",
        "live training: farthest x_pos after each chunk",
    )
    plot_series(
        [r["decisions"] for r in hist],
        [r["loss"] for r in hist],
        "decisions",
        "smooth L1",
        "loss is prediction vs target, not gameplay quality",
    )
elif MARIO_TRAIN.get("mode") != "live":
    story = fallback_paths()["storyboard"]
    if story.is_file():
        display(SVG(filename=str(story)))
    rows = MARIO_TRAIN.get("fallback_metrics") or load_fallback_metrics()
    if rows:
        plot_series(
            [r["decisions"] for r in rows],
            [r["x_pos"] for r in rows],
            "decisions",
            "x_pos",
            "fallback: recorded World 1-1 distance",
        )
print(
    "what just happened: a chunked Mario DQN train, or the recorded-clip fallback. "
    "The next cell lets a labeled checkpoint play for a user-set step budget."
)
'''
    )

    md(
        """\
**Watch the pretrained DQN play Mario.** By default, the next cell loads the public final checkpoint, runs it greedily in World 1-1, renders the gameplay video, and reports reward, distance, and deaths. Switch to `local-trained` only when you want to compare your workshop run. The public checkpoint is educational and may not finish the level.
"""
    )

    code(
        r'''# @title Watch the pretrained DQN play Mario
EVAL_SOURCE = "public-final"  # @param ["public-final", "local-trained"]
EVAL_STEPS = 10000             # @param {type:"integer"}

from pathlib import Path

from IPython.display import SVG, Video, display
from lab import clamp_int, require_names, scorecard
from mario_lab import EVAL_STEPS_RANGE, evaluate_mario, fallback_paths, load_fallback_metrics

require_names(globals(), "HOME", "SEED")
MARIO_MODEL_REPO = globals().get("MARIO_MODEL_REPO") or "BLANK/mario-dqn-workshop"
MARIO_MODEL_REVISION = globals().get("MARIO_MODEL_REVISION") or "main"
steps = clamp_int(EVAL_STEPS, EVAL_STEPS_RANGE[0], EVAL_STEPS_RANGE[1], "EVAL_STEPS")
MARIO_EVAL = evaluate_mario(
    Path(HOME) / "mario-demo",
    eval_source=EVAL_SOURCE,
    eval_steps=steps,
    seed=SEED,
    repo_id=MARIO_MODEL_REPO,
    revision=MARIO_MODEL_REVISION,
    collect_frames=True,
)
scorecard(
    "Mario evaluation",
    [
        ("source", f"{MARIO_EVAL.get('eval_source')} · {MARIO_EVAL.get('checkpoint_source')}"),
        ("mode", MARIO_EVAL.get("mode")),
        ("steps", MARIO_EVAL.get("eval_steps")),
        ("total reward", MARIO_EVAL.get("total_reward")),
        ("farthest distance", MARIO_EVAL.get("farthest_distance")),
        ("deaths", MARIO_EVAL.get("deaths")),
        ("completed episodes", MARIO_EVAL.get("completed_episodes")),
        ("best attempt", MARIO_EVAL.get("best_attempt")),
        ("video", MARIO_EVAL.get("video")),
    ],
)
if MARIO_EVAL.get("video"):
    display(Video(MARIO_EVAL["video"], embed=True, width=360))
elif MARIO_EVAL.get("frames"):
    for i, frame in enumerate(MARIO_EVAL["frames"][:8]):
        show_frame(frame, f"{EVAL_SOURCE} frame {i}", f"mario-eval-{i}")
else:
    story = fallback_paths()["storyboard"]
    if story.is_file():
        display(SVG(filename=str(story)))
    rows = MARIO_EVAL.get("fallback_metrics") or load_fallback_metrics()
    if rows:
        plot_series(
            [r["decisions"] for r in rows],
            [r["x_pos"] for r in rows],
            "decisions",
            "x_pos",
            "fallback: recorded World 1-1 distance",
        )
print(
    "what just happened: a labeled policy played for your step budget. "
    "Next, the same algorithm trains the Atari RAM-vector teacher."
)
'''
    )

    md(
        """\
**Same algorithm, different eyes.** Mario sees pixels through a CNN. The workshop teacher sees a RAM vector through an MLP. Replay, target net, and ε-greedy stay the same. Super Mario Bros is how we *see* the loop. The RAM-vector teacher is still the pipeline.
"""
    )
    diagram_cell("dqn-encoders", "Pixels/CNN vs RAM-vector/MLP — same DQN loop")

    md(
        """\
**Raw homework.** The teacher’s captured transitions are synthetic traces — not training pairs yet. Chapter 6 curates them. First, train the Atari teacher on the critical path.
"""
    )

    challenge(
        "how long should the teacher train?",
        "0 keeps the MODE default (QUICK = 5000). Raise it only if you have time — "
        "FULL at 300k is a real train. QUICK is clipped to 8000.",
    )

    code(
        r'''# @title Teacher knobs
DQN_DECISIONS_OVERRIDE = 0     # @param {type:"integer"}

import json
from pathlib import Path

import torch

from slm_rl.agents.bots import RandomAgent
from slm_rl.teachers import make_teacher
from slm_rl.teachers.dqn import metrics_path_for, train_dqn
from slm_rl.teachers.dqn_checkpoint import expected_dqn_checkpoint
from slm_rl.rollout.runner import EpisodeRunner
from lab import bound, clamp_int, require_names, scorecard

require_names(globals(), "GAME", "HOME", "MODE", "SEED", "RUN_NAME", "DQN_DECISIONS")
game, game_cfg = ensure_game()
decisions = DQN_DECISIONS
if DQN_DECISIONS_OVERRIDE:
    lo, hi = bound(MODE, "dqn_decisions")
    decisions = clamp_int(DQN_DECISIONS_OVERRIDE, int(lo), int(hi), "DQN_DECISIONS_OVERRIDE")
    if MODE == "FULL":
        print("FULL teacher train can take a long time on a T4.")

dqn_device = "cuda" if torch.cuda.is_available() else "cpu"
dqn_path = expected_dqn_checkpoint(GAME, HOME)
print(f"training DQN for {decisions} decisions on {dqn_device} → {dqn_path}")
DQN_SUMMARY = train_dqn(
    game_cfg,
    decisions=decisions,
    out_path=dqn_path,
    device=dqn_device,
    seed=SEED,
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

teacher_agent, teacher_id = make_teacher(game_cfg, seed=SEED, dqn_checkpoint=str(dqn_path))
print(f"make_teacher → model_id={teacher_id!r}")

RANDOM_STATS = globals().get("RANDOM_STATS")
if not RANDOM_STATS:
    unwrap_game(game)
    RANDOM_STATS = EpisodeRunner(
        game, RandomAgent(seed=SEED), game_cfg,
        run_id=RUN_NAME, generation=0, model_id="random",
    ).run_episode(seed=SEED, episode_id="random-000")

unwrap_game(game)
stream_episode(game, every=4)
TEACHER_STATS = EpisodeRunner(
    game, teacher_agent, game_cfg,
    run_id=RUN_NAME, generation=0, model_id=teacher_id,
).run_episode(seed=SEED + 3, episode_id="teacher-000")
unwrap_game(game)

def _rew(stats):
    if not stats:
        return float("nan")
    return float(stats.get("cum_reward") or 0.0)

_random = globals().get("RANDOM_STATS") or {}
_llm = globals().get("LLM_STATS") or {}
scorecard(
    "same GameConfig, three agents",
    [
        ("random", f"{_random.get('outcome')!r}  reward={_rew(_random):.3f}"),
        ("LLM preview", f"{_llm.get('outcome')!r}  reward={_rew(_llm):.3f}  (short)"),
        ("teacher", f"{TEACHER_STATS.get('outcome')!r}  reward={_rew(TEACHER_STATS):.3f}"),
        ("dqn decisions", decisions),
        ("checkpoint", dqn_path),
    ],
)
print(
    "what just happened: a CleanRL-pattern DQN trained on RAM vectors, its "
    "reward curve was plotted from the sibling metrics JSONL, and the teacher "
    "played one streamed episode for comparison."
)
'''
    )

    checkpoint("teacher", "`dqn_path`, `TEACHER_STATS`, optional `MARIO_TRAIN` / `MARIO_EVAL`", "bake a shareable pack")


def chapter_6() -> None:
    chapter_open(
        6,
        "The traces from Chapter 5 are still rollout rows — reward, monitor flags, "
        "sometimes a `Q-values rank…` rationale. They are not training pairs yet.\n\n"
        "This chapter inspects one real row, runs `select_episodes` / `export_sft_dataset`, "
        "then shows the prompt/completion pair. After that, the pack cell bakes a shareable textbook. "
        "Gen-1 SFT is a warm start: it teaches the action alphabet. The eval gate comes later.\n\n"
        "The pack cell reuses Chapter 5's checkpoint (`dqn_decisions=0`). Push is **off** unless you tick the form and set a repo.",
    )
    diagram_cell("trace-to-pair", "DQN play → JSONL row → select → ACTION pair")
    md(
        """\
**Look before you bake.** A raw JSONL row is a decision trace. An SFT pair is `{prompt, completion}`. The next cell runs the same `select_episodes` / `export_sft_dataset` APIs the trainer uses, so you can see the cut.
"""
    )
    code(
        r'''# @title Inspect a trace before SFT
import json
from pathlib import Path

from slm_rl.datagen.sft_export import export_sft_dataset, group_episodes, select_episodes
from slm_rl.datagen.writer import RolloutWriter
from slm_rl.rollout.runner import EpisodeRunner
from lab import require_names, scorecard

require_names(globals(), "HOME", "RUN_NAME", "SEED", "cfg")

raw_path = Path(HOME) / RUN_NAME / "teacher-traces.jsonl"
source = "missing"
if "teacher_agent" in globals() and "teacher_id" in globals():
    game, game_cfg = ensure_game()
    unwrap_game(game)
    with RolloutWriter(raw_path) as writer:
        EpisodeRunner(
            game, teacher_agent, game_cfg, writer=writer,
            run_id=RUN_NAME, generation=0, model_id=teacher_id,
        ).run_episode(seed=SEED + 3, episode_id="teacher-trace-000")
    unwrap_game(game)
    source = "chapter-5-teacher"
elif "jsonl_path" in globals() and Path(jsonl_path).is_file():
    raw_path = Path(jsonl_path)
    source = "chapter-4-jsonl"
else:
    raise RuntimeError("Run Chapter 4 or the Chapter 5 teacher cell first so a JSONL trace exists.")

with raw_path.open(encoding="utf-8") as fh:
    raw = json.loads(next(line for line in fh if line.strip()))

print("=== raw rollout row (not a training pair) ===")
print("keys:", sorted(raw))
completion = raw.get("completion") or raw.get("raw_completion") or ""
scorecard(
    "raw trace",
    [
        ("source", source),
        ("path", raw_path),
        ("episode_id", raw.get("episode_id")),
        ("step_idx", raw.get("step_idx")),
        ("parsed_action", raw.get("parsed_action")),
        ("reward", raw.get("reward")),
        ("model_id", raw.get("model_id")),
        ("completion", str(completion)[:220]),
        ("training pair yet?", "no — still a rollout row"),
    ],
)

n_raw = len(group_episodes(raw_path))
selected = select_episodes(raw_path, cfg.train)
sft_preview = Path(HOME) / RUN_NAME / "sft-preview.jsonl"
n_pairs = export_sft_dataset(raw_path, sft_preview, cfg.train)
print(f"select_episodes: {n_raw} episode(s) in → {len(selected)} kept")
print(f"export_sft_dataset: {n_pairs} prompt/completion pair(s)")

if n_pairs == 0:
    print("no SFT pairs — every step was fallback_random or the episode was dropped.")
else:
    with sft_preview.open(encoding="utf-8") as fh:
        pair = json.loads(next(line for line in fh if line.strip()))
    print("=== after select_episodes / SFT conversion (this is a training pair) ===")
    print("keys:", sorted(pair))
    scorecard(
        "SFT pair",
        [
            ("keys", list(pair)),
            ("prompt roles", [m.get("role") for m in (pair.get("prompt") or [])]),
            ("completion", pair.get("completion")),
        ],
    )
print(
    "what just happened: a raw trace is not a training pair. "
    "select_episodes keeps useful episodes; export_sft_dataset writes prompt → ACTION."
)
'''
    )
    md(
        """\
**What that training export just did** (`sft_export.py`):

1. Drop monitor-flagged doom loops.
2. Keep wins and the top return quantile.
3. Cap identical action sequences.
4. Strip `Q-values rank…` so the student copies `ACTION: RIGHT`, not a fake ranking.

The pack bake below applies its own top-return filter before storing rollouts. SFT export later applies the complete selection and conversion above. SFT teaches the action alphabet. The frozen eval gate decides later — not in this chapter.
"""
    )
    diagram_cell("packs", "bake_pack → disk → Hugging Face → resolve_pack")

    code(
        r'''# @title Packs (optional hub)
PACK_URL = ""          # @param {type:"string"}
PUSH_TO_HUB = False    # @param {type:"boolean"}
PUSH_REPO = ""         # @param {type:"string"}

from pathlib import Path

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
from lab import ask_hf_token, require_names, scorecard

require_names(globals(), "GAME", "HOME", "MODE")

print("ATARI_GAMES:", sorted(ATARI_GAMES))
print(f"is_atari({GAME!r}):", is_atari(GAME))
print("packs_root:", packs_root(HOME))

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

if PACK_URL.strip():
    cached = resolve_pack(PACK_URL.strip(), HOME, GAME)
    print("resolve_pack →", cached)
else:
    print("PACK_URL is empty — local bake only. Paste a dataset URL to pull a published pack.")

if PUSH_TO_HUB and PUSH_REPO.strip():
    token = apply_hf_token(ask_hf_token(default=hf_token() or ""))
    if not token:
        print(
            "No HF token. Paste one in Join the room (or this prompt) and re-run to push. "
            "Nothing was uploaded."
        )
    else:
        commit_url = push_pack(pack_dir, PUSH_REPO.strip(), token=token)
        print("push_pack →", commit_url)
elif PUSH_TO_HUB:
    print("PUSH_TO_HUB is on but PUSH_REPO is empty — nothing uploaded.")
else:
    print("Push is off. Local pack stays on disk.")

scorecard(
    "pack",
    [
        ("dir", pack_dir),
        ("episodes", BAKE_EPISODES),
        ("PACK_URL", PACK_URL or "(none)"),
        ("pushed", bool(PUSH_TO_HUB and PUSH_REPO.strip())),
    ],
)
print(
    f"what just happened: baked a local {GAME} pack at {pack_dir} "
    f"({BAKE_EPISODES} teacher episodes, reused dqn.pt)."
)
'''
    )
    checkpoint("packs", "`pack_dir` + MANIFEST.json", "export SFT/GRPO rows and train")


def chapter_7() -> None:
    chapter_open(
        7,
        "Same factory, two strategies. Both write a LoRA adapter (a few MB, not the whole 1.2B).\n\n"
        "SFT = learn from your best games. GRPO = sample, score, nudge — slower, needs `GameConfig`. "
        "`q4` → QLoRA. T4 compute = fp16. QUICK default trains **reject_sft** only; pick `both` if you want the GRPO comparison.",
    )
    diagram_cell("train-strategies", "parquet feeds reject_sft and GRPO; both write adapter/")
    deck_image("GRPO.png", "Several attempts. Score the group. Prefer better-than-siblings.")
    md(
        f"""\
**RLVR** — we do not need the perfect move. We need to know whether the outcome was better.

| Hit / destroy **+** | Better score **+** |
|---|---|
| {_img_tag(f"{DECK_RAW}/enemy_destroyed.png", "Enemy destroyed", _THUMB_STYLE)} | {_img_tag(f"{DECK_RAW}/better_score.png", "Better score", _THUMB_STYLE)} |
| Life lost / invalid **−** | Doom loop / parse fail **−** |
| {_img_tag(f"{DECK_RAW}/life_lost.png", "Life lost", _THUMB_STYLE)} | {_img_tag(f"{DECK_RAW}/doom_loop.png", "Doom loop", _THUMB_STYLE)} |
"""
    )
    md("**Before & after** — same seeds, raw vs trained. What changed?")
    deck_video("base-0.mp4", "Raw model — Gen 0 baseline.")
    deck_video("RL-trained.mp4", "After RL — valid actions, less spray, a hint of strategy.")

    challenge(
        "which trainer?",
        "Type `reject_sft` (minutes), `grpo` (slower), or `both` (VRAM flush between). "
        "`Runtime → Run all` waits here so you pick the strategy, not the default.",
    )

    code(
        r'''# @title Training strategy
TRAIN_STRATEGY = "reject_sft"   # @param ["reject_sft", "grpo", "both"]

import json
from pathlib import Path

import torch

from slm_rl.datagen.grpo_export import export_grpo_dataset
from slm_rl.datagen.sft_export import export_sft_dataset
from slm_rl.training.lora import release_trainer_memory
from lab import ask, require_names, resolve_choice, scorecard, TRAIN_STRATEGIES

require_names(globals(), "HOME", "RUN_NAME", "cfg", "game_cfg", "BACKEND", "tier")
TRAIN_STRATEGY = ask(
    "Which trainer? reject_sft is minutes; grpo is slower; both needs a VRAM flush",
    allowed=TRAIN_STRATEGIES,
    default=TRAIN_STRATEGY,
)
TRAIN_STRATEGY = resolve_choice(TRAIN_STRATEGY, TRAIN_STRATEGIES, "reject_sft")

# Free the Chapter 3 inference backend before we load a trainable copy.
close_backend_if_any("backend")
release_trainer_memory(torch.cuda.is_available())

model_id = cfg.model or tier.model
dataset_path = parquet_path if Path(parquet_path).is_file() else rollout_dir
if not Path(dataset_path).exists():
    raise FileNotFoundError("Chapter 4 dataset is missing — re-run that cell.")

train_dir = Path(HOME) / RUN_NAME / "train"
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
from lab import scorecard

four_bit = (BACKEND or tier.backend) == "transformers-4bit"
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

class _Skip:
    adapter_path = None
    metrics = {"skipped": True}

sft_out = Path(HOME) / RUN_NAME / "sft"
if TRAIN_STRATEGY not in ("reject_sft", "both"):
    print("Skipping reject_sft (TRAIN_STRATEGY is grpo).")
    SFT_RESULT = _Skip()
else:
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

if torch.cuda.is_available() and TRAIN_STRATEGY in ("reject_sft", "both"):
    print(f"peak VRAM after reject_sft: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

if "sft_strategy" in globals():
    del sft_strategy
release_trainer_memory(torch.cuda.is_available())
scorecard("reject_sft", [("adapter", SFT_RESULT.adapter_path), ("metrics", SFT_RESULT.metrics)])
print(
    "what just happened: reject_sft fitted a LoRA adapter on the best "
    "prompt/completion pairs (or skipped), then released trainer memory."
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
from lab import scorecard

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

class _Skip:
    adapter_path = None
    metrics = {"skipped": True}

grpo_out = Path(HOME) / RUN_NAME / "grpo"
if TRAIN_STRATEGY not in ("grpo", "both"):
    print("Skipping GRPO (TRAIN_STRATEGY is reject_sft). Set it to grpo or both to compare.")
    GRPO_RESULT = _Skip()
else:
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

if "grpo_strategy" in globals():
    del grpo_strategy
release_trainer_memory(torch.cuda.is_available())

ADAPTER_PATH = None
for result in (GRPO_RESULT, SFT_RESULT):
    p = getattr(result, "adapter_path", None)
    if p and Path(p).is_dir():
        ADAPTER_PATH = Path(p)
        break
scorecard(
    "train",
    [
        ("strategy", TRAIN_STRATEGY),
        ("SFT adapter", getattr(SFT_RESULT, "adapter_path", None)),
        ("GRPO adapter", getattr(GRPO_RESULT, "adapter_path", None)),
        ("ADAPTER_PATH", ADAPTER_PATH),
        ("bf16_ok", bf16_ok()),
        ("compute_dtype", compute_dtype(True)),
        ("four_bit", four_bit),
    ],
)
print(
    "what just happened: the selected strategy wrote an adapter (or skipped). "
    "Trainer memory is released again."
)
'''
    )
    checkpoint("train", "`ADAPTER_PATH` (LoRA) + export JSONL", "eval + gate")


def chapter_8() -> None:
    chapter_open(
        8,
        "Same frozen seeds, no teacher, no pruner. Promote only if the SLM itself got better.\n\n"
        "Guards: `min_improvement`, `max_invalid_rate`, `max_intervention_rate_ratio`, `min_mean_entropy`. "
        "Boxing primary = `mean_score`. The cell also sabotages a copy so you see a real \"no\".",
    )
    diagram_cell("eval-gate", "Frozen eval suite → EvalGate.decide → promote or reject")

    challenge(
        "will the gate promote?",
        "Type promote or reject **before** the suite runs. QUICK adapters rarely beat the base. "
        "Then try a harsher `MARGIN_OVERRIDE` on a **copy** of the gate — the real `cfg.gate` stays put.",
    )

    code(
        r'''# @title Gate experiment
PREDICT_GATE = "reject"     # @param ["promote", "reject"]
MARGIN_OVERRIDE = 0.01      # @param {type:"number"}

from slm_rl.agents.llm_agent import LLMAgent
from slm_rl.eval.gate import EvalGate
from slm_rl.eval.suites import run_suite
from slm_rl.games.registry import get_game
from slm_rl.inference.base import GenParams, create_backend
from slm_rl.training.lora import release_trainer_memory
from lab import ask, bound, clamp_float, ensure_card, grade, record_guess, require_names, scorecard

require_names(globals(), "GAME", "MODE", "cfg", "EVAL_LIMIT")
CARD = ensure_card(globals())
PREDICT_GATE = ask(
    "Will the gate promote this adapter, or reject it?",
    allowed=("promote", "reject"),
    default=PREDICT_GATE,
)
game, game_cfg = ensure_game()
lo, hi = bound(MODE, "min_improvement")
MARGIN_OVERRIDE = clamp_float(MARGIN_OVERRIDE, lo, hi, "MARGIN_OVERRIDE")

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

ADAPTER_PATH = globals().get("ADAPTER_PATH")
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

tweaked = cfg.gate.model_copy(update={"min_improvement": MARGIN_OVERRIDE})
tweaked_promote, tweaked_reason = EvalGate(tweaked).decide(BASE_METRICS, CANDIDATE_METRICS)

worse = dict(CANDIDATE_METRICS)
worse["primary"] = float(CANDIDATE_METRICS.get("primary") or 0.0) - 1.0
worse["invalid_rate"] = max(float(CANDIDATE_METRICS.get("invalid_rate") or 0.0), cfg.gate.max_invalid_rate + 0.1)
fake_promote, fake_reason = gate.decide(BASE_METRICS, worse)

scorecard(
    "gate",
    [
        ("your guess", grade(PREDICT_GATE, "promote" if promote else "reject")),
        ("real gate", f"promote={promote}  {reason}"),
        ("copied margin", f"{MARGIN_OVERRIDE} → promote={tweaked_promote}  {tweaked_reason}"),
        ("sabotaged copy", f"promote={fake_promote}  {fake_reason}"),
    ],
)
record_guess(CARD, "gate", PREDICT_GATE, "promote" if promote else "reject")
print(
    "what just happened: the frozen suite was played twice, the gate judged "
    "the real adapter, a copied threshold was applied, then a sabotaged copy was rejected."
)
'''
    )
    checkpoint("eval", "`BASE_METRICS`, `CANDIDATE_METRICS`, gate reason", "run the evolve loop")


def chapter_9() -> None:
    chapter_open(
        9,
        "One generation = one pass. Promotion moves the champion pointer; reject leaves it put.\n\n"
        "`GenerationRunner` reloads game YAML (Boxing = 2500 turns / 100 evals). The cell writes a `config_dir` overlay so QUICK stays short. "
        "`ensure_baseline()` is gen 0, cached.\n\n"
        "QUICK may **not** improve. That's the gate working, not a bug.",
    )
    diagram_cell("evolve-loop", "Same evolve loop as the title card — now you have run the pieces")

    challenge(
        "how many generations?",
        "0 keeps the MODE default. QUICK is capped at 2. A reject is a successful demo of the gate — not a failed workshop.",
    )

    code(
        r'''# @title Evolve knobs
EVOLVE_GENERATIONS = 0     # @param {type:"integer"}

import json
from pathlib import Path

import yaml

from slm_rl.orchestrator.generation import GenerationRunner
from slm_rl.orchestrator.paths import RunPaths
from slm_rl.training.lora import release_trainer_memory
from lab import bound, clamp_int, require_names, scorecard

require_names(globals(), "HOME", "RUN_NAME", "MODE", "cfg", "GAME", "EVAL_LIMIT")
close_backend_if_any("backend")
release_trainer_memory(torch.cuda.is_available())

if EVOLVE_GENERATIONS:
    lo, hi = bound(MODE, "generations")
    cfg = cfg.model_copy(update={"generations": clamp_int(EVOLVE_GENERATIONS, int(lo), int(hi), "EVOLVE_GENERATIONS")})

WORKSHOP_CONFIG_DIR = Path(HOME) / RUN_NAME / "configs"
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

rows = [("generations requested", cfg.generations), ("champion", evolve_runner.registry.champion)]
for g, m in EVOLVE_METRICS:
    gate = m.get("gate") or {}
    ev = m.get("eval") or {}
    rows.append((f"gen {g}", f"promoted={gate.get('promoted')}  primary={ev.get('primary')}  {gate.get('reason')}"))
scorecard("evolve", rows)
print(
    "what just happened: GenerationRunner ran baseline + "
    f"{len(EVOLVE_METRICS)} generation(s). Promotion moves the champion pointer; "
    "rejection leaves it put."
)
'''
    )
    checkpoint("evolve", "`EVOLVE_METRICS` + registry.json", "theater: base vs champion")


def chapter_10() -> None:
    chapter_open(
        10,
        "Payoff: base vs champion on the **same** seeds. Exhibition, not eval — eval is never written to disk.\n\n"
        "`run_exhibition` loads one model at a time. We replay JSONL — no second load.",
    )
    diagram_cell("theater", "Shared exhibition seeds → JSONL → side-by-side replay")

    challenge(
        "pick a rematch seed",
        "Exhibition seeds start at 20_000 so they never collide with eval. "
        "`REPLAY_EVERY` is how often the stacked frame updates (1 = every step).",
    )

    code(
        r'''# @title Theater knobs
THEATER_SEED = 20000     # @param {type:"integer"}
REPLAY_EVERY = 4         # @param {type:"slider", min:1, max:8, step:1}

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image as PILImage, ImageDraw

from slm_rl.games.base import ActionSpec
from slm_rl.games.registry import get_game
from slm_rl.orchestrator.paths import RunPaths
from slm_rl.theater.exhibition import run_exhibition
from slm_rl.training.lora import release_trainer_memory
from lab import bound, clamp_int, require_names, scorecard

require_names(globals(), "cfg", "GAME", "MODE", "WORKSHOP_CONFIG_DIR")
close_backend_if_any("backend")
release_trainer_memory(torch.cuda.is_available())

lo_e, hi_e = bound(MODE, "theater_episodes")
lo_r, hi_r = bound(MODE, "replay_every")
THEATER_EPISODES = clamp_int({"QUICK": 1, "FULL": 3}[MODE], int(lo_e), int(hi_e), "THEATER_EPISODES")
REPLAY_EVERY = clamp_int(REPLAY_EVERY, int(lo_r), int(hi_r), "REPLAY_EVERY")
THEATER_SEED = clamp_int(THEATER_SEED, 20_000, 90_000, "THEATER_SEED")

run_dir = RunPaths(cfg.home, cfg.run_id).root
print(f"run_dir={run_dir}  episodes={THEATER_EPISODES}  seed_start={THEATER_SEED}")

EXHIBITION = run_exhibition(
    run_dir, GAME,
    episodes=THEATER_EPISODES,
    seed_start=THEATER_SEED,
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
    seed_b = int(base_steps[0]["seed"]) if base_steps else THEATER_SEED
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
        if i % REPLAY_EVERY == 0 or i + 1 == n:
            show_frame(
                _compose(ale_rgb(g_base), ale_rgb(g_champ), score_b, score_c, i + 1),
                "theater: base (left) vs champion (right)",
                "theater-ab",
            )

scorecard(
    "theater",
    [
        ("seed_start", THEATER_SEED),
        ("episodes", THEATER_EPISODES),
        ("champion_generation", EXHIBITION.champion_generation),
        ("mean base", f"{_mean_score(base_eps):.3f}"),
        ("mean champion", f"{_mean_score(champ_eps):.3f}"),
        ("message", EXHIBITION.message),
    ],
)
print(
    "what just happened: run_exhibition wrote paired JSONL; we replayed the "
    "recorded actions in two fresh games and hstacked the ALE screens."
)
'''
    )
    checkpoint("theater", "base vs champion replay + scorecard", "optional Hugging Face publish")


def chapter_11() -> None:
    chapter_open(
        11,
        "Paste a write-scoped Hugging Face token in the form (or at Join the room). "
        "Missing token = friendly no-op, never a crash.",
    )
    diagram_cell("publish", "publish_experiment writes a model repo and a dataset repo")

    challenge(
        "publish? (opt-in)",
        "Leave `PUBLISH` unchecked unless you have a write-scoped Hugging Face token. "
        "A missing token is a friendly no-op, never a crash.",
    )

    code(
        r'''# @title Publish to Hugging Face
PUBLISH = False     # @param {type:"boolean"}
HF_TOKEN = ""      # @param {type:"string"}

from pathlib import Path

from slm_rl.datagen.hf_publish import publish_experiment
from slm_rl.hf_auth import apply_hf_token, hf_token
from slm_rl.orchestrator.paths import RunPaths
from lab import ask_hf_token, scorecard

token = None
PUBLISH_RESULT = None
if PUBLISH:
    seeded = (HF_TOKEN or "").strip() or (hf_token() or "")
    token = apply_hf_token(ask_hf_token(default=seeded))

if not PUBLISH:
    print("PUBLISH is off — nothing uploaded. Tick the box and re-run to opt in.")
elif not token:
    print(
        "No HF token. Paste a write-scoped token in the form above "
        "(https://huggingface.co/settings/tokens) and re-run this cell. "
        "Nothing was uploaded."
    )
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

scorecard("publish", [("opted in", PUBLISH), ("result", PUBLISH_RESULT)])
print(
    "what just happened: "
    + (
        "publish_experiment uploaded (or reported a partial failure on) the run."
        if PUBLISH_RESULT is not None
        else "publish was skipped."
    )
)
'''
    )

    md(
        """\
**Share the win.** After a successful publish, copy the LinkedIn post below. On LinkedIn, turn **Agentics Foundation** and **Ideas2IT** into `@` company mentions (company pages: [agentics-org](https://www.linkedin.com/company/agentics-org), [ideas2it](https://www.linkedin.com/company/ideas2it)).
"""
    )

    code(
        r'''# @title LinkedIn post (copy after publish)
result = globals().get("PUBLISH_RESULT")
model_url = "https://huggingface.co/{username}/slm-rl-{run_id}"
dataset_url = "https://huggingface.co/datasets/{username}/slm-rl-{run_id}"
if result is not None:
    if getattr(result, "model_repo", None):
        model_url = f"https://huggingface.co/{result.model_repo}"
    if getattr(result, "dataset_repo", None):
        dataset_url = f"https://huggingface.co/datasets/{result.dataset_repo}"

LINKEDIN_POST = f"""Just published my first SLM-RL run to Hugging Face.

Trained a small language model with reinforcement learning on Atari — rollout → train → eval → gate — then pushed the adapter and dataset to the Hub.

Proud to have been part of training and publishing with the first SLM-RL workshop, in association with Agentics Foundation and Ideas2IT.

Model: {model_url}
Dataset: {dataset_url}

#SLM #ReinforcementLearning #HuggingFace #AgenticAI"""

print(LINKEDIN_POST)
print()
if result is None or not (getattr(result, "model_repo", None) or getattr(result, "dataset_repo", None)):
    print(
        "URLs still have placeholders — publish successfully in the cell above, "
        "then re-run this cell to fill them in."
    )
else:
    print("Copy everything above the blank line into LinkedIn, then @-mention the two companies.")
'''
    )
    checkpoint("publish", "HF repos + LinkedIn post if opted in, else a no-op", "register your own game")


def chapter_12() -> None:
    chapter_open(
        12,
        "Pure Python, seed-deterministic, no ML imports. Required: `reset`, `step`, `state_hash`, `system_prompt`, `eval_suite`.\n\n"
        "The cell registers `guess-number` and rolls it out with the same runner as Boxing. "
        "Tune the rewards in the form — harsher misses change how a random agent looks.",
    )
    diagram_cell("game-abc", "Game ABC → registry → the same rollout / train / eval path")

    challenge(
        "design a tiny reward",
        "Miss penalty default is `-0.1`. Try `-0.5` (harsher) or `0` (only the win matters). "
        "Then compare the random-agent total.",
    )

    code(
        r'''# @title Your game tweak
MISS_REWARD = -0.1     # @param {type:"number"}
WIN_REWARD = 1.0       # @param {type:"number"}

import hashlib
import random
from pathlib import Path

from slm_rl.agents.bots import RandomAgent
from slm_rl.config.schema import GameConfig
from slm_rl.datagen.writer import RolloutWriter
from slm_rl.eval.suites import EvalSuite
from slm_rl.games.base import ActionSpec, Game, Observation, StepResult
from slm_rl.games.registry import available_games, get_game, register_game
from slm_rl.rollout.runner import EpisodeRunner
from lab import bound, clamp_float, require_names, scorecard

require_names(globals(), "HOME", "RUN_NAME", "MODE", "SEED")
lo_m, hi_m = bound(MODE, "miss_reward")
lo_w, hi_w = bound(MODE, "win_reward")
MISS_REWARD = clamp_float(MISS_REWARD, lo_m, hi_m, "MISS_REWARD")
WIN_REWARD = clamp_float(WIN_REWARD, lo_w, hi_w, "WIN_REWARD")


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
            return StepResult(self._obs("Correct."), WIN_REWARD, True, False, {"outcome": "win"})
        if guess < self._secret:
            self._low = max(self._low, guess + 1)
            hint = "too low"
        else:
            self._high = min(self._high, guess - 1)
            hint = "too high"
        truncated = self._turn >= self.config.max_turns
        text = f"{guess} is {hint}. Range is now {self._low}-{self._high}."
        info = {"outcome": "loss"} if truncated else {}
        return StepResult(self._obs(text), MISS_REWARD, False, truncated, info)

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
guess_path = Path(HOME) / RUN_NAME / "guess-number.jsonl"
with RolloutWriter(guess_path) as writer:
    GUESS_STATS = EpisodeRunner(
        guess_game, RandomAgent(seed=SEED), guess_cfg, writer=writer,
        run_id=RUN_NAME, generation=0, model_id="random",
    ).run_episode(seed=SEED, episode_id="guess-000")
scorecard(
    "guess-number",
    [
        ("win_reward", WIN_REWARD),
        ("miss_reward", MISS_REWARD),
        ("outcome", GUESS_STATS.get("outcome")),
        ("cum_reward", GUESS_STATS.get("cum_reward")),
        ("steps", GUESS_STATS.get("steps")),
        ("jsonl", guess_path),
    ],
)
print(
    "what just happened: a brand-new game was registered in-process and "
    "rolled out with the same EpisodeRunner. The pipeline did not change."
)
'''
    )

    optional_challenge(
        "reference solution",
        "A miss should be a small negative (`-0.1`) so the agent is not indifferent, "
        "and a win should be `+1.0` so `eval_suite` `win_rate` still lines up with reward. "
        "To ship the game outside the notebook, add:\n\n"
        "```toml\n"
        "[project.entry-points.\"slm_rl.games\"]\n"
        "guess-number = \"my_pkg.guess:GuessNumberGame\"\n"
        "```",
    )
    md(
        """\
Nothing else changed — everything speaks the `Game` ABC.
"""
    )
    checkpoint(
        "plugin game",
        "`guess-number` registered + one EpisodeRunner rollout",
        "run the fast CPU test slice",
        chapter=12,
    )


def chapter_13() -> None:
    chapter_open(
        13,
        "Fast slice: Boxing, config merge, JSONL writer. Not the full suite.",
    )
    deck_image("meme_never_move.png", "Reward hacking: the agent learns what is scored — not what you meant.")
    deck_image("doom.png", "Doom loop: same thought, same action, progress dies while the episode continues.")

    code(
        r'''!python -m pytest -q -x tests/test_boxing.py tests/test_config.py tests/test_rollout_writer.py
print("what just happened: a fast subset of the repo tests ran and stopped on the first failure if any.")
'''
    )

    code(
        r'''from lab import ensure_card, show_card

CARD = ensure_card(globals())
show_card(CARD)
print("that card is yours — Runtime → Run all cannot fill it in.")
'''
    )

    md(
        """\
## Appendix

Skipped on purpose: Vue playground / Docker, MLX, vLLM, vision backend, hosted ELO.

Docs: `ARCHITECTURE` · `PIPELINE` · `HYBRID_RL` · `PLUGIN_GUIDE` · `LIFECYCLE`

T4-heavy cells that need a Colab GPU: Chapter 3 (model download), 5 (DQN), 7 (LoRA), 8 (eval), 9 (evolve), 10 (theater). CPU can still run Chapters 0–2, 4's schema bits, 12, 13, and the builder tests. Mario is optional and falls back without a GPU.

Talk track (deck on one screen, this notebook on the other):

| Colab | Presentation slides |
|---|---|
| 0 Setup | cover → join-lobby → why-slm-matters → today |
| 1 Games | what-is-slm → why-games · play before you train |
| 2 Config | notebook-led; deck stays on the story |
| 3 Model plays | journey → journey-tech |
| 4 Dataset | rl-loop → sft-vs-rl → why-sft-first |
| 5 Teachers | section-dqn → what-is-dqn → q-values → Bellman → replay → target → ε → clips → live train → eval → bridge → teacher-dataset |
| 6 Packs | synthetic-homework → inspect a trace → trace-to-pair → dataset-filters → why-warmstart |
| 7 Training | gen-0-1 → GRPO slides |
| 8 Gate | promote-reject → eval-gate |
| 9 Evolve | self-improve → champion-rollouts |
| 10 Theater | improvement |
| 12 Your game | beyond-atari |

| Symptom | Fix |
|---|---|
| CUDA OOM | `PRECISION=q4`, `MODE=QUICK`, `close_backend_if_any()` |
| bitsandbytes fails | `PRECISION=fp16` |
| Session died | Re-run Ch. 0 + knobs, or uncomment Drive mount |
| Slow first load | ~2 GB download; cached after that |
| Changed GAME | Re-run knobs through Chapter 1 (`ensure_game` rebuilds) |
| Form value ignored | Re-run that yellow cell — later cells read the names as-is |
| Stuck at a text prompt | Type a non-blank answer — `Runtime → Run all` is paused on purpose. |
"""
    )
    checkpoint(
        "workshop",
        "a run folder under `HOME`/`RUN_NAME` plus whatever you published",
        "screenshot theater or the evolve scorecard for the honor-system tournament",
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
    global _CURRENT_CHAPTER
    cells.clear()
    _CURRENT_CHAPTER = None
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
