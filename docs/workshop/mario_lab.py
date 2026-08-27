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

# Optional hosted checkpoint. Empty URL means "no download; use fallback
# unless a local file is supplied." Architecture is Nature-style DQN.
CHECKPOINT_URL = ""
CHECKPOINT_SHA256 = ""
CHECKPOINT_N_ACTIONS = 2  # RIGHT, RIGHT+A
ACTION_NAMES = ("RIGHT", "RIGHT+A")
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
LEARNING_RATE = 2.5e-4

HERE = Path(__file__).resolve().parent
FALLBACK_DIR = HERE / "assets" / "mario"


def pinned_packages() -> tuple[str, ...]:
    if sys.version_info >= (3, 13):
        return MODERN_PACKAGES
    return LEGACY_PACKAGES


def fallback_paths() -> dict[str, Path]:
    return {
        "storyboard": FALLBACK_DIR / "fallback_storyboard.svg",
        "metrics": FALLBACK_DIR / "fallback_metrics.jsonl",
        "notes": FALLBACK_DIR / "README.md",
    }


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


def download_checkpoint(dest: Path, url: str = CHECKPOINT_URL) -> Path | None:
    if not url:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url, timeout=30) as resp, dest.open("wb") as fh:
            fh.write(resp.read())
    except Exception as exc:  # noqa: BLE001 — gated demo
        print(f"checkpoint download failed: {exc}")
        return None
    if not verify_checkpoint(dest):
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


def try_make_mario_env():
    """Return (env, err). env is None on any failure."""
    try:
        import gym_super_mario_bros
        from gym_super_mario_bros.actions import RIGHT_ONLY
        from nes_py.wrappers import JoypadSpace
    except Exception as exc:  # noqa: BLE001
        return None, f"import failed: {exc}"
    try:
        if hasattr(gym_super_mario_bros, "make"):
            env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
        else:
            import gym

            env = gym.make("SuperMarioBros-1-1-v0")
        env = JoypadSpace(env, RIGHT_ONLY)
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
        "encoder": "pixels → CNN",
        "teacher_encoder": "RAM vector → MLP",
        "shared": ["replay buffer", "Bellman target", "target net", "epsilon-greedy"],
    }

    ckpt = checkpoint
    if ckpt is None and CHECKPOINT_URL:
        ckpt = download_checkpoint(workdir / "mario_dqn.pt")
    if ckpt is not None and not verify_checkpoint(Path(ckpt)):
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
            payload = torch.load(ckpt, map_location=device, weights_only=False)
            state = payload.get("online") or payload.get("state_dict") or payload
            q_net.load_state_dict(state)
            result["checkpoint"] = str(ckpt)
        target.load_state_dict(q_net.state_dict())
        target.eval()
        opt = torch.optim.Adam(q_net.parameters(), lr=LEARNING_RATE)

        obs = _reset(env)
        frames = [preprocess_frame(obs)]
        replay: list[tuple] = []
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed)

        for step in range(play_steps):
            stacked = stack_frames(frames)
            tensor = torch.tensor(stacked, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q = q_net(tensor)[0]
            result["q_values"] = [round(float(v), 3) for v in q.tolist()]
            action = int(torch.argmax(q).item())
            obs, reward, done, info = _step(env, action)
            result["rewards"].append(reward)
            result["x_pos"].append(int((info or {}).get("x_pos") or 0))
            nxt = preprocess_frame(obs)
            replay.append((stacked, action, reward, stack_frames(frames + [nxt]), done))
            frames.append(nxt)
            if done:
                obs = _reset(env)
                frames = [preprocess_frame(obs)]

        # Bounded continuation: learn from the just-collected replay. This is
        # a short update, not a claim that we resumed a full trainer state.
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
