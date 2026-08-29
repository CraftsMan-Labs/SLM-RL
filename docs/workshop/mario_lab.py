"""Optional Mario DQN demo for the Colab workshop.

Isolated from the SLM-RL teacher path. The repository teacher is a RAM-vector
MLP (``slm_rl.teachers.dqn``). This module is a pixel CNN on Super Mario
Bros. used only to teach DQN intuition.

The live path is gated: missing deps, a failed checksum, or an emulator
crash return ``mode="fallback"`` and never raise into later chapters.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable
from urllib.request import urlopen

# Pinned for Colab (usually 3.11/3.12) vs modern CPython (3.13+).
LEGACY_PACKAGES = (
    "gym==0.25.2",
    "gym-super-mario-bros==7.4.0",
    "nes-py>=8.2.1,<9",
)
MODERN_PACKAGES = (
    "gym-super-mario-bros==9.1.0",
    "nes-py>=9.0.1",
)

# Public Nature-style CNN checkpoints. Prefer the workshop Hugging Face
# repo; GitHub staged files remain the verified fallback.
DEFAULT_MARIO_MODEL_REPO = "BLANK/mario-dqn-workshop"
DEFAULT_MARIO_MODEL_REVISION = "main"
CHECKPOINT_SOURCE = {
    "repo": DEFAULT_MARIO_MODEL_REPO,
    "fallback_repo": "https://github.com/alanfrancis442/mario-ai",
    "architecture": "Nature DQN CNN, 4×84×84 → RIGHT / RIGHT+A",
    "license": "educational weights; Super Mario Bros is owned by Nintendo",
}
LOCAL_TRAINED_NAME = "local-trained.chkpt"
TRAIN_MINUTES_RANGE = (1.0, 20.0)
EVAL_STEPS_RANGE = (100, 50_000)
EVAL_INTERVAL_RANGE = (50, 2_000)
DEFAULT_TRAIN_MINUTES = 15.0
DEFAULT_EVAL_STEPS = 10_000
DEFAULT_EVAL_INTERVAL = 400
BUFFER_CAPACITY = 20_000
BATCH_SIZE = 32
TARGET_SYNC_EVERY = 250
MIN_REPLAY = 200
EPS_DECAY_DECISIONS = 50_000
CPU_TRAIN_MINUTES = 5.0

STAGED_CHECKPOINTS: tuple[dict[str, Any], ...] = (
    {
        "stage": "untrained",
        "label": "Untrained",
        "note": "falls in the first pit",
        "url": "",
        "sha256": "",
        "filename": "",
        "decisions": 0,
        "clip": "mario-untrained.mp4",
    },
    {
        "stage": "mid",
        "label": "Mid",
        "note": "holds right, clears a pipe",
        "url": (
            "https://raw.githubusercontent.com/alanfrancis442/mario-ai/master/"
            "checkpoints/2025-10-12T00-21-50/mario_net_0.chkpt"
        ),
        "sha256": "4cbe70ceb7783337cc1bb72f63aeddd06c3e7f31e9baedb62ed937b6321dd03b",
        "filename": "mario_mid.chkpt",
        "decisions": 20000,
        "clip": "mario-mid.mp4",
    },
    {
        "stage": "pretrained",
        "label": "Pretrained",
        "note": "clears the first pit and reaches the first pipe",
        "url": (
            "https://raw.githubusercontent.com/alanfrancis442/mario-ai/master/"
            "checkpoints/2025-11-03T00-19-18/mario_net_0.chkpt"
        ),
        "sha256": "75521a9b48c792693bf855ec5c62dfaf8bda1a2bf556439c674bfb36fde1c4e5",
        "filename": "mario_pretrained.chkpt",
        "decisions": 250000,
        "clip": "mario-pretrained.mp4",
    },
)

# Named workshop checkpoints. SHA-256 matches the GitHub staged files so
# the same blobs can be uploaded to Hugging Face without relabeling.
HF_CHECKPOINTS: dict[str, dict[str, Any]] = {
    "warm-start": {
        "filename": "warm-start.chkpt",
        "sha256": STAGED_CHECKPOINTS[1]["sha256"],
        "fallback_stage": "mid",
        "note": "early World 1-1 play; workshop live-training default",
    },
    "final": {
        "filename": "final.chkpt",
        "sha256": STAGED_CHECKPOINTS[2]["sha256"],
        "fallback_stage": "pretrained",
        "note": "later World 1-1 play; public evaluation default",
    },
}

# Optional hosted checkpoint for the live Colab cell. Empty means "use the
# pretrained staged file if already downloaded, else random weights."
CHECKPOINT_URL = STAGED_CHECKPOINTS[-1]["url"]
CHECKPOINT_SHA256 = STAGED_CHECKPOINTS[-1]["sha256"]
CHECKPOINT_N_ACTIONS = 2  # RIGHT, RIGHT+A
ACTION_NAMES = ("RIGHT", "RIGHT+A")
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
LEARNING_RATE = 2.5e-4
SKIP_FRAMES = 4
RECORD_FPS = 30
CLIP_FILES = tuple(row["clip"] for row in STAGED_CHECKPOINTS)

HERE = Path(__file__).resolve().parent
FALLBACK_DIR = HERE / "assets" / "mario"
MANIFEST_NAME = "clip_manifest.json"

# Sequential indices from the public MarioNet checkpoint → our Q-net.
_ONLINE_TO_QNET = {
    "0": "conv.0",
    "2": "conv.2",
    "4": "conv.4",
    "7": "head.1",
    "9": "head.3",
}


def pinned_packages() -> tuple[str, ...]:
    if sys.version_info >= (3, 13):
        return MODERN_PACKAGES
    return LEGACY_PACKAGES


def ensure_mario_packages(*, install: bool = False) -> tuple[bool, str]:
    """Return (ok, reason). pip only runs when ``install`` is True."""
    env, err = try_make_mario_env()
    if env is not None:
        try:
            env.close()
        except Exception:
            pass
        return True, "already installed"
    if not install:
        return False, err or "mario env unavailable"
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", *pinned_packages()]
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"pip install failed: {exc}"
    env, err = try_make_mario_env()
    if env is None:
        return False, err or "still unavailable after install"
    try:
        env.close()
    except Exception:
        pass
    return True, "installed"


def staged_checkpoint(stage: str) -> dict[str, Any]:
    for row in STAGED_CHECKPOINTS:
        if row["stage"] == stage:
            return row
    raise KeyError(stage)


def fallback_paths() -> dict[str, Path]:
    paths = {
        "storyboard": FALLBACK_DIR / "fallback_storyboard.svg",
        "metrics": FALLBACK_DIR / "fallback_metrics.jsonl",
        "notes": FALLBACK_DIR / "README.md",
        "manifest": FALLBACK_DIR / MANIFEST_NAME,
    }
    for name in CLIP_FILES:
        paths[Path(name).stem] = FALLBACK_DIR / name
    return paths


def clip_paths() -> dict[str, Path]:
    return {row["stage"]: FALLBACK_DIR / row["clip"] for row in STAGED_CHECKPOINTS}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: Path, expected: str = CHECKPOINT_SHA256) -> bool:
    if not path.is_file():
        return False
    if not expected:
        return True
    return sha256_file(path) == expected


def download_checkpoint(dest: Path, url: str = CHECKPOINT_URL, *, sha256: str = "") -> Path | None:
    if not url:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url, timeout=120) as resp, dest.open("wb") as fh:
            fh.write(resp.read())
    except Exception as exc:  # noqa: BLE001 — gated demo
        print(f"checkpoint download failed: {exc}")
        return None
    expected = sha256 or CHECKPOINT_SHA256
    if expected and not verify_checkpoint(dest, expected):
        print("checkpoint checksum mismatch; ignoring file")
        dest.unlink(missing_ok=True)
        return None
    return dest


def make_qnet(n_actions: int = CHECKPOINT_N_ACTIONS):
    """Nature-style CNN: 4×84×84 → n_actions. Torch is imported lazily."""
    import torch.nn as nn

    class MarioQNet(nn.Module):
        def __init__(self, n_actions: int) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU(),
                nn.Linear(512, n_actions),
            )

        def forward(self, x):
            return self.head(self.conv(x))

    return MarioQNet(n_actions)


def remap_online_keys(state: dict[str, Any]) -> dict[str, Any]:
    """Map a Sequential-index CNN (0/2/4/7/9) onto ``conv.*`` / ``head.*``."""
    remapped: dict[str, Any] = {}
    for key, value in state.items():
        if key.startswith("conv.") or key.startswith("head."):
            remapped[key] = value
            continue
        prefix, _, rest = key.partition(".")
        if prefix in _ONLINE_TO_QNET and rest:
            remapped[f"{_ONLINE_TO_QNET[prefix]}.{rest}"] = value
        else:
            remapped[key] = value
    return remapped


def extract_online_state(payload: Any) -> dict[str, Any]:
    """Pull the playable online weights out of a public Mario checkpoint."""
    if not isinstance(payload, dict):
        return {}
    blob: Any = payload
    for key in ("online", "state_dict", "model", "online_model"):
        nested = blob.get(key) if isinstance(blob, dict) else None
        if isinstance(nested, dict):
            blob = nested
            break
    if not isinstance(blob, dict):
        return {}
    if any(str(k).startswith("online.") for k in blob):
        blob = {str(k)[len("online.") :]: v for k, v in blob.items() if str(k).startswith("online.")}
    return remap_online_keys(blob)


def load_qnet_weights(q_net: Any, path: Path) -> str:
    """Load remapped public weights. Returns a short source label."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = extract_online_state(payload)
    if not state:
        if isinstance(payload, dict):
            state = payload.get("online") or payload.get("state_dict") or payload
        else:
            raise ValueError(f"unrecognized checkpoint: {path}")
    q_net.load_state_dict(state)
    return path.name


def preprocess_frame(frame) -> Any:
    """RGB or gray frame → float32 84×84. No torch required."""
    import numpy as np

    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr.mean(axis=-1)
    arr = arr.astype("float32")
    if arr.max() > 1.5:
        arr = arr / 255.0
    # nearest-neighbor resize without extra deps
    h, w = arr.shape[:2]
    ys = (np.linspace(0, h - 1, 84)).astype(int)
    xs = (np.linspace(0, w - 1, 84)).astype(int)
    return arr[ys][:, xs]


def stack_frames(frames: list[Any]) -> Any:
    import numpy as np

    while len(frames) < 4:
        frames = [frames[0]] + frames
    return np.stack(frames[-4:], axis=0)


# Matches the public 2-way MarioNet: not gym's 5-way RIGHT_ONLY (which starts with NOOP).
TWO_ACTIONS = [["right"], ["right", "A"]]


def try_make_mario_env():
    """Return (env, err). env is None on any failure."""
    try:
        import gym_super_mario_bros
        from nes_py.wrappers import JoypadSpace
    except Exception as exc:  # noqa: BLE001
        return None, f"import failed: {exc}"
    try:
        if hasattr(gym_super_mario_bros, "make"):
            env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
        else:
            import gym

            env = gym.make("SuperMarioBros-1-1-v0")
        env = JoypadSpace(env, TWO_ACTIONS)
        return env, ""
    except Exception as exc:  # noqa: BLE001
        return None, f"env make failed: {exc}"


def _reset(env):
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def _step(env, action: int):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, float(reward), done, info
    obs, reward, done, info = out
    return obs, float(reward), bool(done), info


def _step_skip(env, action: int, skip: int = SKIP_FRAMES):
    frames: list[Any] = []
    total = 0.0
    done = False
    info: dict[str, Any] = {}
    obs = None
    for _ in range(max(1, skip)):
        obs, reward, done, info = _step(env, action)
        frames.append(obs)
        total += reward
        if done:
            break
    return obs, total, done, info, frames


def load_fallback_metrics(path: Path | None = None) -> list[dict[str, Any]]:
    metrics_path = path or fallback_paths()["metrics"]
    rows: list[dict[str, Any]] = []
    if not metrics_path.is_file():
        return rows
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_clip_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest = path or fallback_paths()["manifest"]
    if not manifest.is_file():
        return {}
    return json.loads(manifest.read_text(encoding="utf-8"))


def encode_mp4(frames: list[Any], dest: Path, *, fps: int = RECORD_FPS) -> bool:
    """Write RGB frames to a browser-safe H.264 MP4. Returns False on failure."""
    import numpy as np

    if not frames:
        return False
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    first = np.asarray(frames[0])
    if first.ndim != 3 or first.shape[-1] != 3:
        return False
    height, width = first.shape[:2]
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    blob = b"".join(np.asarray(frame, dtype="uint8").tobytes() for frame in frames)
    try:
        proc = subprocess.run(cmd, input=blob, capture_output=True, check=False)
    except OSError as exc:
        print(f"ffmpeg missing: {exc}")
        return False
    if proc.returncode != 0 or not dest.is_file():
        err = (proc.stderr or b"").decode("utf-8", errors="replace")[-400:]
        print(f"ffmpeg failed: {err}")
        return False
    return True


def play_mario_policy(
    env: Any,
    q_net: Any,
    *,
    device: Any,
    decisions: int,
    seed: int = 0,
    skip: int = SKIP_FRAMES,
    collect_frames: bool = False,
) -> dict[str, Any]:
    """Greedy play. Returns rewards, x_pos, last Q-values, optional RGB frames."""
    import numpy as np
    import torch

    obs = _reset(env)
    gray = [preprocess_frame(obs)]
    rgb_frames: list[Any] = [np.asarray(obs)] if collect_frames else []
    rewards: list[float] = []
    x_pos: list[int] = []
    q_values: list[float] = []
    for _ in range(decisions):
        stacked = stack_frames(gray)
        tensor = torch.tensor(stacked, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q = q_net(tensor)[0]
        q_values = [round(float(v), 3) for v in q.tolist()]
        action = int(torch.argmax(q).item())
        obs, reward, done, info, skipped = _step_skip(env, action, skip=skip)
        rewards.append(reward)
        x_pos.append(int((info or {}).get("x_pos") or 0))
        if collect_frames:
            rgb_frames.extend(np.asarray(frame) for frame in skipped)
        nxt = preprocess_frame(obs)
        gray.append(nxt)
        if done:
            break
    return {
        "q_values": q_values,
        "rewards": rewards,
        "x_pos": x_pos,
        "frames": rgb_frames,
        "decisions_taken": len(rewards),
        "max_x_pos": max(x_pos) if x_pos else 0,
        "sum_reward": float(sum(rewards)),
    }


def resolve_staged_checkpoint(row: dict[str, Any], workdir: Path) -> Path | None:
    if not row.get("url"):
        return None
    dest = workdir / (row.get("filename") or f"{row['stage']}.chkpt")
    if dest.is_file() and (not row.get("sha256") or verify_checkpoint(dest, row["sha256"])):
        return dest
    return download_checkpoint(dest, row["url"], sha256=str(row.get("sha256") or ""))


def record_mario_clips(
    dest_dir: Path | None = None,
    *,
    workdir: Path | None = None,
    decisions: int = 180,
    seed: int = 0,
) -> dict[str, Any]:
    """Record untrained / mid / pretrained World 1-1 clips.

    Always returns a result dict. ``mode`` is ``live`` or ``fallback``.
    """
    dest_dir = Path(dest_dir or FALLBACK_DIR)
    workdir = Path(workdir or dest_dir / "checkpoints")
    dest_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "mode": "fallback",
        "reason": "",
        "source": dict(CHECKPOINT_SOURCE),
        "clips": {},
        "metrics": [],
    }

    env, err = try_make_mario_env()
    if env is None:
        result["reason"] = err or "mario env unavailable"
        return result

    try:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metrics: list[dict[str, Any]] = []
        clips: dict[str, str] = {}
        for row in STAGED_CHECKPOINTS:
            try:
                env.close()
            except Exception:
                pass
            env, err = try_make_mario_env()
            if env is None:
                result["reason"] = err or "mario env unavailable"
                return result
            q_net = make_qnet(CHECKPOINT_N_ACTIONS).to(device)
            ckpt = resolve_staged_checkpoint(row, workdir)
            loaded = ""
            if ckpt is not None:
                loaded = load_qnet_weights(q_net, ckpt)
            q_net.eval()
            play = play_mario_policy(
                env,
                q_net,
                device=device,
                decisions=decisions,
                seed=seed,
                collect_frames=True,
            )
            clip_path = dest_dir / row["clip"]
            if not encode_mp4(play["frames"], clip_path):
                result["reason"] = f"encode failed for {row['stage']}"
                return result
            clips[row["stage"]] = str(clip_path)
            metrics.append(
                {
                    "stage": row["stage"],
                    "label": row["label"],
                    "note": row["note"],
                    "decisions": row["decisions"],
                    "decisions_taken": play["decisions_taken"],
                    "x_pos": play["max_x_pos"],
                    "mean_ep_reward": round(play["sum_reward"] / max(play["decisions_taken"], 1), 3),
                    "q_values": play["q_values"],
                    "checkpoint": loaded or "random",
                    "clip": row["clip"],
                    "sha256": sha256_file(clip_path),
                }
            )
        manifest = {
            "source": CHECKPOINT_SOURCE,
            "fps": RECORD_FPS,
            "skip_frames": SKIP_FRAMES,
            "seed": seed,
            "metrics": metrics,
        }
        (dest_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        result.update({"mode": "live", "reason": "recorded World 1-1 clips", "clips": clips, "metrics": metrics})
        return result
    except Exception as exc:  # noqa: BLE001
        result["reason"] = f"record failed: {exc}"
        return result
    finally:
        try:
            env.close()
        except Exception:
            pass


def run_mario_demo(
    workdir: Path,
    *,
    play_steps: int = 400,
    continue_steps: int = 200,
    checkpoint: Path | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Play a pretrained (or random) Mario DQN, then a short learning update.

    Always returns a result dict. ``mode`` is ``live`` or ``fallback``.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "mode": "fallback",
        "reason": "",
        "action_names": list(ACTION_NAMES),
        "q_values": [],
        "rewards": [],
        "losses": [],
        "x_pos": [],
        "checkpoint": None,
        "fallback": {k: str(v) for k, v in fallback_paths().items()},
        "clips": {k: str(v) for k, v in clip_paths().items() if v.is_file()},
        "encoder": "pixels → CNN",
        "teacher_encoder": "RAM vector → MLP",
        "shared": ["replay buffer", "Bellman target", "target net", "epsilon-greedy"],
        "formula": "target = reward + γ × best next Q",
        "gamma": GAMMA,
    }

    ckpt = checkpoint
    if ckpt is None:
        pretrained = STAGED_CHECKPOINTS[-1]
        ckpt = resolve_staged_checkpoint(pretrained, workdir)
    if ckpt is not None and pretrained_hash_mismatch(ckpt):
        ckpt = None

    env, err = try_make_mario_env()
    if env is None:
        result["reason"] = err or "mario env unavailable"
        result["fallback_metrics"] = load_fallback_metrics()
        return result

    try:
        import torch
        import torch.nn.functional as F

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        q_net = make_qnet(CHECKPOINT_N_ACTIONS).to(device)
        target = make_qnet(CHECKPOINT_N_ACTIONS).to(device)
        if ckpt is not None:
            result["checkpoint"] = load_qnet_weights(q_net, Path(ckpt))
        target.load_state_dict(q_net.state_dict())
        target.eval()
        opt = torch.optim.Adam(q_net.parameters(), lr=LEARNING_RATE)

        play = play_mario_policy(
            env,
            q_net,
            device=device,
            decisions=max(1, play_steps // max(SKIP_FRAMES, 1)),
            seed=seed,
            collect_frames=False,
        )
        result["q_values"] = play["q_values"]
        result["rewards"] = play["rewards"]
        result["x_pos"] = play["x_pos"]

        # Bounded continuation on a tiny replay of the just-played frames.
        # Rebuild a short buffer by playing again with the same net.
        obs = _reset(env)
        frames = [preprocess_frame(obs)]
        replay: list[tuple] = []
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed)
        for _ in range(min(64, play_steps)):
            stacked = stack_frames(frames)
            tensor = torch.tensor(stacked, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q = q_net(tensor)[0]
            action = int(torch.argmax(q).item())
            obs, reward, done, _info, _skipped = _step_skip(env, action)
            nxt = preprocess_frame(obs)
            replay.append((stacked, action, reward, stack_frames(frames + [nxt]), done))
            frames.append(nxt)
            if done:
                obs = _reset(env)
                frames = [preprocess_frame(obs)]

        q_net.train()
        batch_n = min(32, len(replay))
        for i in range(continue_steps):
            if batch_n < 8:
                break
            idx = torch.randint(0, len(replay), (batch_n,), generator=rng)
            batch = [replay[int(j)] for j in idx.tolist()]
            obs_b = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=device)
            act_b = torch.tensor([b[1] for b in batch], dtype=torch.int64, device=device)
            rew_b = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
            nxt_b = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=device)
            done_b = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device)
            q = q_net(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = target(nxt_b).max(1).values
                target_q = rew_b + GAMMA * next_q * (1.0 - done_b)
            loss = F.smooth_l1_loss(q, target_q)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % 20 == 0:
                result["losses"].append(round(float(loss.item()), 4))
            if i and i % 100 == 0:
                target.load_state_dict(q_net.state_dict())

        result["mode"] = "live"
        result["reason"] = (
            "inference + bounded Bellman update"
            if ckpt
            else "untrained CNN + bounded Bellman update (no pretrained checkpoint)"
        )
        result["play_steps"] = play_steps
        result["continue_steps"] = continue_steps
        result["device"] = str(device)
        return result
    except Exception as exc:  # noqa: BLE001
        result["mode"] = "fallback"
        result["reason"] = f"live demo failed: {exc}"
        result["fallback_metrics"] = load_fallback_metrics()
        return result
    finally:
        try:
            env.close()
        except Exception:
            pass


def pretrained_hash_mismatch(path: Path) -> bool:
    row = STAGED_CHECKPOINTS[-1]
    expected = str(row.get("sha256") or "")
    if not expected:
        return False
    return not verify_checkpoint(Path(path), expected)


def _clamp_float(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(lo, min(hi, parsed))


def _clamp_int(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(lo, min(hi, parsed))


def hf_checkpoint_spec(name: str) -> dict[str, Any]:
    if name not in HF_CHECKPOINTS:
        raise KeyError(name)
    return dict(HF_CHECKPOINTS[name])


def download_hf_checkpoint(
    dest: Path,
    *,
    repo_id: str = DEFAULT_MARIO_MODEL_REPO,
    filename: str,
    revision: str = DEFAULT_MARIO_MODEL_REVISION,
    sha256: str = "",
) -> Path | None:
    """Download one public file, pin a revision, and verify SHA-256."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and (not sha256 or verify_checkpoint(dest, sha256)):
        return dest
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # noqa: BLE001
        print(f"huggingface_hub unavailable: {exc}")
        return None
    try:
        cached = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            token=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"HF download failed ({repo_id}@{revision}/{filename}): {exc}")
        return None
    src = Path(cached)
    if not src.is_file():
        return None
    if sha256 and not verify_checkpoint(src, sha256):
        print("HF checkpoint checksum mismatch; ignoring file")
        return None
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return dest


def resolve_named_checkpoint(
    name: str,
    workdir: Path,
    *,
    repo_id: str = DEFAULT_MARIO_MODEL_REPO,
    revision: str = DEFAULT_MARIO_MODEL_REVISION,
) -> tuple[Path | None, str]:
    """Resolve warm-start / final via HF, then the staged GitHub fallback."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if name == "local-trained":
        local = workdir / LOCAL_TRAINED_NAME
        if local.is_file():
            return local, "local-trained"
        return None, "local-trained checkpoint missing"
    spec = hf_checkpoint_spec(name)
    dest = workdir / spec["filename"]
    hf = download_hf_checkpoint(
        dest,
        repo_id=repo_id,
        filename=spec["filename"],
        revision=revision,
        sha256=str(spec.get("sha256") or ""),
    )
    if hf is not None:
        return hf, f"hf:{repo_id}/{spec['filename']}@{revision}"
    fallback = staged_checkpoint(str(spec["fallback_stage"]))
    path = resolve_staged_checkpoint(fallback, workdir)
    if path is None:
        return None, f"{name} unavailable (HF and GitHub fallback failed)"
    return path, f"github-fallback:{fallback['stage']}"


def save_training_checkpoint(
    path: Path,
    q_net: Any,
    target: Any,
    optimizer: Any,
    meta: dict[str, Any],
) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "online": q_net.state_dict(),
            "target": target.state_dict(),
            "optimizer": optimizer.state_dict(),
            "meta": dict(meta),
        },
        path,
    )
    return path


def load_training_checkpoint(
    q_net: Any,
    path: Path,
    *,
    target: Any = None,
    optimizer: Any = None,
) -> dict[str, Any]:
    import torch

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "online" in payload and "meta" in payload:
        q_net.load_state_dict(payload["online"])
        if target is not None and payload.get("target") is not None:
            target.load_state_dict(payload["target"])
        if optimizer is not None and payload.get("optimizer") is not None:
            try:
                optimizer.load_state_dict(payload["optimizer"])
            except Exception:
                pass
        return dict(payload.get("meta") or {})
    load_qnet_weights(q_net, Path(path))
    return {}


def maybe_copy_to_drive(src: Path, *, enabled: bool) -> str:
    if not enabled:
        return ""
    drive = Path("/content/drive/MyDrive/slm-rl-mario")
    if not drive.parent.parent.is_dir():
        return "Drive not mounted"
    try:
        drive.mkdir(parents=True, exist_ok=True)
        dest = drive / src.name
        shutil.copy2(src, dest)
        return str(dest)
    except Exception as exc:  # noqa: BLE001
        return f"Drive copy failed: {exc}"


def epsilon_at(decisions: int) -> float:
    t = min(1.0, max(0, decisions) / float(EPS_DECAY_DECISIONS))
    return EPS_START + (EPS_END - EPS_START) * t


def _device_and_budget(train_minutes: float) -> tuple[Any, float, str]:
    import torch

    minutes = _clamp_float(train_minutes, *TRAIN_MINUTES_RANGE, DEFAULT_TRAIN_MINUTES)
    if torch.cuda.is_available():
        return torch.device("cuda"), minutes, "cuda"
    return torch.device("cpu"), min(minutes, CPU_TRAIN_MINUTES), "cpu"


def train_mario_live(
    workdir: Path,
    *,
    training_mode: str = "warm-start",
    train_minutes: float = DEFAULT_TRAIN_MINUTES,
    eval_interval: int = DEFAULT_EVAL_INTERVAL,
    seed: int = 0,
    save_to_drive: bool = False,
    repo_id: str = DEFAULT_MARIO_MODEL_REPO,
    revision: str = DEFAULT_MARIO_MODEL_REVISION,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    collect_eval_frames: bool = False,
    max_decisions: int | None = None,
    eval_decisions: int = 80,
) -> dict[str, Any]:
    """Chunked DQN updates. Always returns a result dict."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    mode = "warm-start" if str(training_mode).strip().lower() != "from-scratch" else "from-scratch"
    interval = _clamp_int(eval_interval, *EVAL_INTERVAL_RANGE, DEFAULT_EVAL_INTERVAL)
    result: dict[str, Any] = {
        "mode": "fallback",
        "reason": "",
        "training_mode": mode,
        "history": [],
        "checkpoint": None,
        "drive_copy": "",
        "action_names": list(ACTION_NAMES),
        "encoder": "pixels → CNN",
        "teacher_encoder": "RAM vector → MLP",
        "fallback": {k: str(v) for k, v in fallback_paths().items()},
        "clips": {k: str(v) for k, v in clip_paths().items() if v.is_file()},
    }

    env, err = try_make_mario_env()
    if env is None:
        result["reason"] = err or "mario env unavailable"
        result["fallback_metrics"] = load_fallback_metrics()
        return result

    try:
        import torch
        import torch.nn.functional as F

        device, minutes, device_name = _device_and_budget(train_minutes)
        result["device"] = device_name
        result["train_minutes"] = minutes
        q_net = make_qnet(CHECKPOINT_N_ACTIONS).to(device)
        target = make_qnet(CHECKPOINT_N_ACTIONS).to(device)
        opt = torch.optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
        decisions = 0
        source = "from-scratch"
        ckpt: Path | None = workdir / LOCAL_TRAINED_NAME
        if ckpt.is_file():
            meta = load_training_checkpoint(q_net, ckpt, target=target, optimizer=opt)
            decisions = int(meta.get("decisions") or 0)
            source = "local-resume"
        elif mode == "warm-start":
            ckpt, source = resolve_named_checkpoint(
                "warm-start", workdir, repo_id=repo_id, revision=revision
            )
            if ckpt is not None:
                load_training_checkpoint(q_net, ckpt)
                decisions = int(staged_checkpoint("mid")["decisions"])
            else:
                result["reason"] = source
                result["fallback_metrics"] = load_fallback_metrics()
                return result
        target.load_state_dict(q_net.state_dict())
        target.eval()

        rng = torch.Generator(device="cpu")
        rng.manual_seed(int(seed))
        replay: deque[tuple[Any, int, float, Any, bool]] = deque(maxlen=BUFFER_CAPACITY)
        deadline = time.monotonic() + minutes * 60.0
        obs = _reset(env)
        frames = [preprocess_frame(obs)]
        deaths = 0
        last_loss = 0.0
        history: list[dict[str, Any]] = []
        next_eval = interval
        chunk_reward = 0.0

        def evaluate_now() -> dict[str, Any]:
            play = play_mario_policy(
                env,
                q_net,
                device=device,
                decisions=max(1, int(eval_decisions)),
                seed=seed,
                collect_frames=collect_eval_frames,
            )
            return {
                "max_x_pos": play["max_x_pos"],
                "sum_reward": play["sum_reward"],
                "decisions_taken": play["decisions_taken"],
                "q_values": play["q_values"],
                "frames": play.get("frames") or [],
            }

        q_net.train()
        while time.monotonic() < deadline:
            if max_decisions is not None and decisions >= max_decisions:
                break
            stacked = stack_frames(frames)
            tensor = torch.tensor(stacked, dtype=torch.float32, device=device).unsqueeze(0)
            eps = epsilon_at(decisions)
            if float(torch.rand((), generator=rng)) < eps:
                action = int(torch.randint(0, CHECKPOINT_N_ACTIONS, (1,), generator=rng).item())
            else:
                with torch.no_grad():
                    action = int(torch.argmax(q_net(tensor)[0]).item())
            nxt_obs, reward, done, _info, _skipped = _step_skip(env, action)
            nxt = preprocess_frame(nxt_obs)
            replay.append((stacked, action, float(reward), stack_frames(frames + [nxt]), bool(done)))
            frames.append(nxt)
            chunk_reward += float(reward)
            decisions += 1
            if done:
                deaths += 1
                obs = _reset(env)
                frames = [preprocess_frame(obs)]

            if len(replay) >= MIN_REPLAY:
                idx = torch.randint(0, len(replay), (min(BATCH_SIZE, len(replay)),), generator=rng)
                batch = [replay[int(j)] for j in idx.tolist()]
                obs_b = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=device)
                act_b = torch.tensor([b[1] for b in batch], dtype=torch.int64, device=device)
                rew_b = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
                nxt_b = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=device)
                done_b = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device)
                q = q_net(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target(nxt_b).max(1).values
                    target_q = rew_b + GAMMA * next_q * (1.0 - done_b)
                loss = F.smooth_l1_loss(q, target_q)
                opt.zero_grad()
                loss.backward()
                opt.step()
                last_loss = float(loss.item())
                if decisions % TARGET_SYNC_EVERY == 0:
                    target.load_state_dict(q_net.state_dict())

            if decisions >= next_eval or time.monotonic() >= deadline:
                eval_row = evaluate_now()
                row = {
                    "decisions": decisions,
                    "epsilon": round(epsilon_at(decisions), 4),
                    "loss": round(last_loss, 4),
                    "chunk_reward": round(chunk_reward, 3),
                    "deaths": deaths,
                    "eval": {
                        "max_x_pos": eval_row["max_x_pos"],
                        "sum_reward": eval_row["sum_reward"],
                        "decisions_taken": eval_row["decisions_taken"],
                        "q_values": eval_row["q_values"],
                    },
                }
                history.append(row)
                if on_progress is not None:
                    payload = dict(row)
                    payload["frames"] = eval_row.get("frames") or []
                    on_progress(payload)
                chunk_reward = 0.0
                next_eval = decisions + interval
                obs = _reset(env)
                frames = [preprocess_frame(obs)]

        local = save_training_checkpoint(
            workdir / LOCAL_TRAINED_NAME,
            q_net,
            target,
            opt,
            {"decisions": decisions, "epsilon": epsilon_at(decisions), "training_mode": mode},
        )
        result.update(
            {
                "mode": "live",
                "reason": f"{mode} train on {device_name} ({source})",
                "source": source,
                "history": history,
                "checkpoint": str(local),
                "decisions": decisions,
                "deaths": deaths,
                "drive_copy": maybe_copy_to_drive(local, enabled=save_to_drive),
            }
        )
        return result
    except Exception as exc:  # noqa: BLE001
        result["mode"] = "fallback"
        result["reason"] = f"live train failed: {exc}"
        result["fallback_metrics"] = load_fallback_metrics()
        return result
    finally:
        try:
            env.close()
        except Exception:
            pass


def evaluate_mario(
    workdir: Path,
    *,
    eval_source: str = "local-trained",
    eval_steps: int = DEFAULT_EVAL_STEPS,
    seed: int = 0,
    repo_id: str = DEFAULT_MARIO_MODEL_REPO,
    revision: str = DEFAULT_MARIO_MODEL_REVISION,
    collect_frames: bool = True,
    frame_stride: int = 4,
    on_frame: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    """Greedy evaluation. ``eval_steps`` is a raw emulator-frame budget."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    steps = _clamp_int(eval_steps, *EVAL_STEPS_RANGE, DEFAULT_EVAL_STEPS)
    source_name = "public-final" if str(eval_source).strip().lower() == "public-final" else "local-trained"
    result: dict[str, Any] = {
        "mode": "fallback",
        "reason": "",
        "eval_source": source_name,
        "eval_steps": steps,
        "total_reward": 0.0,
        "farthest_distance": 0,
        "deaths": 0,
        "completed_episodes": 0,
        "best_attempt": 0,
        "episodes": [],
        "video": None,
        "frames": [],
        "fallback": {k: str(v) for k, v in fallback_paths().items()},
        "clips": {k: str(v) for k, v in clip_paths().items() if v.is_file()},
    }

    if source_name == "local-trained":
        ckpt, source = resolve_named_checkpoint("local-trained", workdir)
        if ckpt is None:
            ckpt, source = resolve_named_checkpoint(
                "warm-start", workdir, repo_id=repo_id, revision=revision
            )
    else:
        ckpt, source = resolve_named_checkpoint(
            "final", workdir, repo_id=repo_id, revision=revision
        )
    result["checkpoint_source"] = source
    if ckpt is None:
        result["reason"] = source
        result["fallback_metrics"] = load_fallback_metrics()
        return result

    env, err = try_make_mario_env()
    if env is None:
        result["reason"] = err or "mario env unavailable"
        result["fallback_metrics"] = load_fallback_metrics()
        return result

    try:
        import numpy as np
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        q_net = make_qnet(CHECKPOINT_N_ACTIONS).to(device)
        load_training_checkpoint(q_net, ckpt)
        q_net.eval()

        remaining = steps
        episodes: list[dict[str, Any]] = []
        rgb_frames: list[Any] = []
        total_reward = 0.0
        farthest = 0
        deaths = 0
        while remaining > 0:
            decisions = max(1, remaining // max(SKIP_FRAMES, 1))
            play = play_mario_policy(
                env,
                q_net,
                device=device,
                decisions=decisions,
                seed=seed + len(episodes),
                collect_frames=collect_frames,
            )
            used = play["decisions_taken"] * SKIP_FRAMES
            remaining -= max(used, 1)
            total_reward += float(play["sum_reward"])
            farthest = max(farthest, int(play["max_x_pos"]))
            deaths += 1
            episodes.append(
                {
                    "reward": play["sum_reward"],
                    "max_x_pos": play["max_x_pos"],
                    "decisions_taken": play["decisions_taken"],
                    "q_values": play["q_values"],
                }
            )
            if collect_frames:
                for i, frame in enumerate(play.get("frames") or []):
                    if i % max(1, frame_stride) != 0:
                        continue
                    arr = np.asarray(frame)
                    rgb_frames.append(arr)
                    if on_frame is not None:
                        on_frame(arr)
            if play["decisions_taken"] < decisions:
                continue
            break

        video_path = workdir / f"eval-{source_name}.mp4"
        encoded = encode_mp4(rgb_frames, video_path) if rgb_frames else False
        best = max((row["max_x_pos"] for row in episodes), default=0)
        result.update(
            {
                "mode": "live",
                "reason": f"evaluated {source_name} from {source}",
                "checkpoint": str(ckpt),
                "total_reward": round(total_reward, 3),
                "farthest_distance": farthest,
                "deaths": deaths,
                "completed_episodes": deaths,
                "best_attempt": best,
                "episodes": episodes,
                "video": str(video_path) if encoded else None,
                "frames": [] if encoded else rgb_frames[:: max(1, len(rgb_frames) // 12 or 1)][:12],
                "device": str(device),
            }
        )
        return result
    except Exception as exc:  # noqa: BLE001
        result["mode"] = "fallback"
        result["reason"] = f"evaluate failed: {exc}"
        result["fallback_metrics"] = load_fallback_metrics()
        return result
    finally:
        try:
            env.close()
        except Exception:
            pass
