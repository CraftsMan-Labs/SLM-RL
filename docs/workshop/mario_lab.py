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
import subprocess
import sys
from pathlib import Path
from typing import Any
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

# Public Nature-style CNN checkpoints from alanfrancis442/mario-ai
# (4×84×84, RIGHT / RIGHT+A). Educational weights; Nintendo owns SMB.
CHECKPOINT_SOURCE = {
    "repo": "https://github.com/alanfrancis442/mario-ai",
    "architecture": "Nature DQN CNN, 4×84×84 → RIGHT / RIGHT+A",
    "license": "educational weights; Super Mario Bros is owned by Nintendo",
}

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
