"""CleanRL-pattern DQN teacher over `Game.vector_obs()` (HYBRID_RL.md build
order item 2): the principled upgrade over the hand-written heuristic teacher
(plan 009) for games with no exact solver. Trains at decision granularity by
driving the `Game` ABC directly (not a raw gym env) -- the DECISION level is
the MDP the teacher (and later, the LLM) acts in.

8GB rule: torch is a training-tier extra, never imported at module scope --
every torch symbol lives behind a function/class body so `import
slm_rl.teachers.dqn` succeeds with no optional extras installed.
"""

from __future__ import annotations

import json
import random
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from slm_rl.agents.base import ActionDecision, Agent
from slm_rl.agents.llm_agent import build_messages
from slm_rl.games.base import Observation

if TYPE_CHECKING:
    from slm_rl.config.schema import GameConfig

# Epsilon-greedy schedule during training (CleanRL dqn.py defaults, D9):
# linear decay over the first half of the run, floor at 5%.
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_FRACTION = 0.5
GAMMA = 0.99
LEARNING_RATE = 2.5e-4
BATCH_SIZE = 128
BUFFER_CAPACITY = 100_000  # float32 obs, ~128 dims: ~100k*128*4*2 (s,s') ~= 100MB, 8GB-safe
TARGET_SYNC_EVERY = 1000  # gradient updates, not decisions
TRAIN_EVERY = 4  # decisions
WARMUP_DECISIONS = 1000
CHECKPOINT_EVERY = 25_000  # decisions; crash tolerance via atomic rename
EVAL_EVERY = 25_000  # greedy hold-out episodes for the monitor "validation" curve
EVAL_EPISODES = 5

# Inference-time exploration (same diversity lesson as plan 009's
# HeuristicInvaderAgent: greedy play on a near-seed-invariant env produces
# duplicate warm-start episodes without it).
EXPLORE_EPS = 0.05


def metrics_path_for(out_path: Path | str) -> Path:
    """Sibling JSONL the playground Teachers page tails (`*.metrics.jsonl`)."""
    p = Path(out_path)
    return p.with_name(p.stem + ".metrics.jsonl")


def _append_metric(metrics_path: Path, row: dict[str, Any]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        **row,
    }
    with metrics_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _eval_greedy(
    game_cls,
    game_cfg: "GameConfig",
    q_net,
    action_ids: list[str],
    device,
    *,
    seed: int,
    episodes: int = EVAL_EPISODES,
) -> dict[str, float | int]:
    """Run greedy episodes on a fresh env (does not touch the train env)."""
    import torch

    n_actions = len(action_ids)
    action_id_to_idx = {aid: i for i, aid in enumerate(action_ids)}
    returns: list[float] = []
    for i in range(episodes):
        game = game_cls(game_cfg)
        obs = game.reset(seed=seed + i)
        ep_r = 0.0
        while True:
            legal_idx = [action_id_to_idx[a.id] for a in obs.legal_actions]
            with torch.no_grad():
                vec = game.vector_obs()
                q = q_net(torch.tensor([vec], dtype=torch.float32, device=device))[0]
                mask = torch.full((n_actions,), float("-inf"))
                for j in legal_idx:
                    mask[j] = 0.0
                action_idx = int(torch.argmax(q.cpu() + mask).item())
            action = next(a for a in obs.legal_actions if a.id == action_ids[action_idx])
            result = game.step(action)
            ep_r += result.reward
            if result.terminated or result.truncated:
                break
            obs = result.observation
        returns.append(ep_r)
    return {
        "episodes": len(returns),
        "mean_ep_reward": sum(returns) / len(returns) if returns else 0.0,
    }


def _make_qnet(obs_dim: int, n_actions: int):
    """Builds the QNet class lazily (torch import stays out of module scope)
    and returns an instance: MLP obs_dim -> 256 -> 256 -> n_actions, ReLU."""
    import torch.nn as nn

    class QNet(nn.Module):
        def __init__(self, obs_dim: int, n_actions: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, n_actions),
            )

        def forward(self, x):
            return self.net(x)

    return QNet(obs_dim, n_actions)


class _ReplayBuffer:
    """Plain deque of float32 transitions -- simplicity over throughput; a
    training run is minutes-to-hours, not a bottleneck worth preallocated
    arrays for. (ponytail: CleanRL's own buffer preallocates numpy arrays;
    a deque is ~100k*128*4 bytes here either way, so the simpler version
    stays 8GB-safe.)"""

    def __init__(self, capacity: int, seed: int):
        self.capacity = capacity
        self.buf: deque = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(self, obs, action_idx: int, reward: float, next_obs, done: bool) -> None:
        self.buf.append((obs, action_idx, reward, next_obs, done))

    def sample(self, batch_size: int):
        return self._rng.sample(self.buf, batch_size)

    def __len__(self) -> int:
        return len(self.buf)


def train_dqn(
    game_cfg: "GameConfig",
    decisions: int,
    out_path: str | Path,
    device: str = "cpu",
    seed: int = 0,
    log_every: int = 500,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.02,
) -> dict:
    """CleanRL-pattern DQN training loop, decision-granularity, driving the
    `Game` object directly (games stay ML-free; this lives in teachers/).
    Returns a summary dict; also the payload written to `out_path`.

    `decisions` is a ceiling. If `early_stop_patience > 0`, training stops
    after that many consecutive EVAL_EVERY windows with no eval-reward
    improvement > `early_stop_min_delta` (fraction of best); the checkpoint is
    written on each new best, so early-stop keeps the peak, not a drifted tail.
    """
    import torch
    import torch.nn.functional as F

    from slm_rl.games.registry import get_game

    out_path = Path(out_path)
    metrics_path = metrics_path_for(out_path)
    game_cls = get_game(game_cfg.name)
    game = game_cls(game_cfg)

    rng = random.Random(seed)
    torch.manual_seed(seed)
    dev = torch.device(device)

    episode_seed = seed
    obs = game.reset(seed=episode_seed)
    obs_dim = len(game.vector_obs())
    action_ids = [a.id for a in obs.legal_actions]
    n_actions = len(action_ids)
    action_id_to_idx = {aid: i for i, aid in enumerate(action_ids)}

    q_net = _make_qnet(obs_dim, n_actions).to(dev)
    target_net = _make_qnet(obs_dim, n_actions).to(dev)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    optimizer = torch.optim.Adam(q_net.parameters(), lr=LEARNING_RATE)

    buffer = _ReplayBuffer(BUFFER_CAPACITY, seed=seed * 10_007 + 1)

    episodes = 0
    updates = 0
    last_loss: float | None = None
    best_eval: float | None = None
    evals_since_best = 0
    stopped_early = False
    ep_reward = 0.0
    ep_rewards: deque = deque(maxlen=20)
    _append_metric(
        metrics_path,
        {
            "split": "meta",
            "game": game_cfg.name,
            "decisions_target": decisions,
            "device": device,
            "seed": seed,
            "out_path": str(out_path),
        },
    )
    print(
        f"[dqn] {game_cfg.name}: decisions={decisions} device={device} "
        f"metrics={metrics_path}",
        flush=True,
    )

    def eps_at(step: int) -> float:
        decay_steps = max(1, int(decisions * EPS_DECAY_FRACTION))
        frac = min(1.0, step / decay_steps)
        return EPS_START + frac * (EPS_END - EPS_START)

    def legal_action_indices(o: Observation) -> list[int]:
        return [action_id_to_idx[a.id] for a in o.legal_actions]

    cur_vec = game.vector_obs()

    for step in range(decisions):
        eps = eps_at(step)
        legal_idx = legal_action_indices(obs)
        if rng.random() < eps:
            action_idx = rng.choice(legal_idx)
        else:
            with torch.no_grad():
                q = q_net(torch.tensor([cur_vec], dtype=torch.float32, device=dev))[0]
                mask = torch.full((n_actions,), float("-inf"))
                for i in legal_idx:
                    mask[i] = 0.0
                action_idx = int(torch.argmax(q.cpu() + mask).item())

        action = next(a for a in obs.legal_actions if a.id == action_ids[action_idx])
        result = game.step(action)
        next_vec = game.vector_obs()
        done = result.terminated or result.truncated

        buffer.push(cur_vec, action_idx, result.reward, next_vec, done)
        ep_reward += result.reward

        if done:
            ep_rewards.append(ep_reward)
            ep_reward = 0.0
            episodes += 1
            episode_seed += 1  # incrementing seed: diversity via no-op starts
            obs = game.reset(seed=episode_seed)
            cur_vec = game.vector_obs()
        else:
            obs = result.observation
            cur_vec = next_vec

        if step >= WARMUP_DECISIONS and step % TRAIN_EVERY == 0 and len(buffer) >= BATCH_SIZE:
            batch = buffer.sample(BATCH_SIZE)
            b_obs = torch.tensor([t[0] for t in batch], dtype=torch.float32, device=dev)
            b_act = torch.tensor([t[1] for t in batch], dtype=torch.long, device=dev)
            b_rew = torch.tensor([t[2] for t in batch], dtype=torch.float32, device=dev)
            b_next = torch.tensor([t[3] for t in batch], dtype=torch.float32, device=dev)
            b_done = torch.tensor([t[4] for t in batch], dtype=torch.float32, device=dev)

            q_values = q_net(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = target_net(b_next).max(dim=1).values
                td_target = b_rew + GAMMA * (1.0 - b_done) * next_q
            loss = F.smooth_l1_loss(q_values, td_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
            updates += 1

            if updates % TARGET_SYNC_EVERY == 0:
                target_net.load_state_dict(q_net.state_dict())

        if (step + 1) % log_every == 0:
            mean_r = sum(ep_rewards) / len(ep_rewards) if ep_rewards else 0.0
            loss_str = f"{last_loss:.4f}" if last_loss is not None else "n/a"
            print(
                f"decisions={step + 1} episodes={episodes} eps={eps:.3f} "
                f"mean_ep_reward_last20={mean_r:.4f} loss={loss_str}",
                flush=True,
            )
            _append_metric(
                metrics_path,
                {
                    "split": "train",
                    "decisions": step + 1,
                    "episodes": episodes,
                    "eps": eps,
                    "mean_ep_reward": mean_r,
                    "loss": last_loss,
                },
            )

        # When early-stop is on we checkpoint on new-best eval only (below), so
        # skip the periodic write that would clobber the peak with a later, worse net.
        if (step + 1) % CHECKPOINT_EVERY == 0 and early_stop_patience == 0:
            _write_checkpoint(
                out_path, q_net, obs_dim, action_ids, game_cfg.name,
                step + 1, ep_rewards,
            )

        if (step + 1) % EVAL_EVERY == 0:
            # Hold-out seeds far from train seeds so the curve is "validation".
            ev = _eval_greedy(
                game_cls, game_cfg, q_net, action_ids, dev,
                seed=1_000_000 + step + 1,
            )
            print(
                f"eval decisions={step + 1} episodes={ev['episodes']} "
                f"mean_ep_reward={ev['mean_ep_reward']:.4f}",
                flush=True,
            )
            _append_metric(
                metrics_path,
                {
                    "split": "eval",
                    "decisions": step + 1,
                    "episodes": ev["episodes"],
                    "mean_ep_reward": ev["mean_ep_reward"],
                },
            )
            if early_stop_patience > 0:
                mean = ev["mean_ep_reward"]
                delta = abs(best_eval) * early_stop_min_delta if best_eval else 0.0
                if best_eval is None or mean > best_eval + delta:
                    best_eval = mean
                    evals_since_best = 0
                    _write_checkpoint(
                        out_path, q_net, obs_dim, action_ids, game_cfg.name,
                        step + 1, ep_rewards,
                    )
                else:
                    evals_since_best += 1
                if evals_since_best >= early_stop_patience:
                    stopped_early = True
                    print(
                        f"[dqn] early stop at decisions={step + 1}: "
                        f"eval flat for {evals_since_best} windows "
                        f"(best mean_ep_reward={best_eval:.4f})",
                        flush=True,
                    )
                    break

    mean_reward_last20 = sum(ep_rewards) / len(ep_rewards) if ep_rewards else 0.0
    # Keep the best-eval checkpoint when early-stop ran and found one; otherwise
    # write the final net (patience=0 legacy path, or no eval fired yet).
    if not (early_stop_patience > 0 and best_eval is not None):
        _write_checkpoint(out_path, q_net, obs_dim, action_ids, game_cfg.name, decisions, ep_rewards)
    _append_metric(
        metrics_path,
        {
            "split": "train",
            "decisions": decisions,
            "episodes": episodes,
            "eps": eps_at(decisions - 1),
            "mean_ep_reward": mean_reward_last20,
            "loss": last_loss,
            "done": True,
            "stopped_early": stopped_early,
            "best_eval": best_eval,
        },
    )

    return {
        "decisions": decisions,
        "episodes": episodes,
        "updates": updates,
        "mean_ep_reward_last20": mean_reward_last20,
        "loss": last_loss,
        "buffer_size": len(buffer),
        "out_path": str(out_path),
        "metrics_path": str(metrics_path),
        "stopped_early": stopped_early,
        "best_eval": best_eval,
    }


def _write_checkpoint(out_path: Path, q_net, obs_dim, action_ids, game_name, decisions, ep_rewards) -> None:
    import torch

    payload = {
        "state_dict": q_net.state_dict(),
        "obs_dim": obs_dim,
        "n_actions": len(action_ids),
        "action_ids": action_ids,
        "game": game_name,
        "decisions": decisions,
        "mean_ep_reward_last20": sum(ep_rewards) / len(ep_rewards) if ep_rewards else 0.0,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(out_path)  # atomic rename: crash tolerance mid-write


class DQNTeacherAgent(Agent):
    """Mirrors `HeuristicInvaderAgent`'s contract exactly: loads a checkpoint
    written by `train_dqn`, plays greedy-with-exploration, and verbalizes an
    honest rationale for either branch (same diversity lesson as plan 009 --
    without EXPLORE_EPS, a greedy DQN on a near-seed-invariant env produces
    duplicate warm-start episodes)."""

    def __init__(self, checkpoint_path: str | Path, system_prompt: str, seed: int | None = None):
        import torch

        self.system_prompt = system_prompt
        self._rng = random.Random(seed)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.action_ids: list[str] = checkpoint["action_ids"]
        self.model_id = f"teacher:{checkpoint['game'].replace('-', '_')}_dqn"
        self._q_net = _make_qnet(checkpoint["obs_dim"], checkpoint["n_actions"])
        self._q_net.load_state_dict(checkpoint["state_dict"])
        self._q_net.eval()

    def act(self, obs: Observation) -> ActionDecision:
        import torch

        if self._rng.random() < EXPLORE_EPS:
            action = self._rng.choice(obs.legal_actions)
            rationale = f"Trying {action.id} to vary my approach."
            completion = f"{rationale}\nACTION: {action.id}"
            return ActionDecision(
                action=action,
                raw_completion=completion,
                prompt_messages=build_messages(self.system_prompt, obs),
            )

        vec = obs.metadata["vector_obs"]
        with torch.no_grad():
            q = self._q_net(torch.tensor([vec], dtype=torch.float32))[0]

        legal_ids = {a.id for a in obs.legal_actions}
        by_id = {a.id: a for a in obs.legal_actions}
        ranked = sorted(
            (
                (aid, float(q[i].item()))
                for i, aid in enumerate(self.action_ids)
                if aid in legal_ids
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )
        best_id, best_q = ranked[0]
        if len(ranked) > 1:
            second_id, second_q = ranked[1]
            rationale = (
                f"Q-values rank {best_id} highest ({best_q:.2f}; next {second_id} {second_q:.2f})."
            )
        else:
            rationale = f"Q-values rank {best_id} highest ({best_q:.2f})."

        action = by_id[best_id]
        completion = f"{rationale}\nACTION: {action.id}"
        return ActionDecision(
            action=action,
            raw_completion=completion,
            prompt_messages=build_messages(self.system_prompt, obs),
        )
