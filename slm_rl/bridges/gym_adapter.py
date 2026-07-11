"""Gymnasium adapter: wraps an external `gymnasium.Env` (ALE with
obs_type="ram") into our `Game` contract via a per-game ObservationRenderer
(RAM -> text). Requires the [atari] extra."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Any

from slm_rl.games.base import ActionSpec, Game, Observation, StepResult


class ObservationRenderer(ABC):
    """Per-game RAM/state -> text + legal actions. Freeway's implementation
    lives in slm_rl/games/atari/ram_maps/freeway.py."""

    @abstractmethod
    def render(self, raw_obs, info: dict) -> tuple[str, list[ActionSpec]]: ...


class GymnasiumGameAdapter(Game):
    """Bridges a Gymnasium/ALE env (RAM observations) into the `Game`
    contract. Concrete per-game subclasses fix `env_id` and `renderer` and
    implement `system_prompt()`/`eval_suite()` (registry entries are
    per-game, not for this adapter).

    8GB rule: `gymnasium`/`ale_py` are imported lazily inside `reset`, never
    at module top level — the core package must still import with no
    optional extras installed.
    """

    def __init__(
        self,
        config,
        opponent=None,
        env_id: str = "",
        renderer: ObservationRenderer | None = None,
    ):
        super().__init__(config, opponent)
        self.env_id = env_id
        self.renderer = renderer
        extra = config.extra
        self.action_repeat: int = int(extra.get("action_repeat", 3))
        self.score_scale: float = float(extra.get("score_scale", 30.0))
        self.life_loss_penalty: float = float(extra.get("life_loss_penalty", -0.5))

        self._env = None
        self._action_ids: list[str] = []
        self._ram = None
        self._info: dict[str, Any] = {}
        self._turn = 0
        self._score = 0.0
        self._lives: int | None = None

    def _ensure_env(self):
        if self._env is None:
            import ale_py
            import gymnasium as gym

            gym.register_envs(ale_py)
            self._env = gym.make(
                self.env_id,
                obs_type="ram",
                frameskip=4,
                repeat_action_probability=0.0,
            )
        return self._env

    def reset(self, seed: int | None = None) -> Observation:
        env = self._ensure_env()
        ram, info = env.reset(seed=seed)
        self._action_ids = list(env.unwrapped.get_action_meanings())
        self._ram = ram
        self._info = info
        self._turn = 0
        self._score = 0.0
        self._lives = info.get("lives")
        return self._observation()

    def step(self, action: ActionSpec) -> StepResult:
        env = self._ensure_env()
        ale_action = self._action_ids.index(action.id)

        raw_sum = 0.0
        terminated = truncated = False
        info: dict[str, Any] = self._info
        ram = self._ram
        for _ in range(self.action_repeat):
            ram, reward, terminated, truncated, info = env.step(ale_action)
            raw_sum += reward
            if terminated or truncated:
                break

        self._score += raw_sum
        reward = raw_sum / self.score_scale

        prev_lives = self._lives
        cur_lives = info.get("lives")
        if prev_lives is not None and cur_lives is not None and cur_lives < prev_lives:
            reward += self.life_loss_penalty  # penalty is negative, so add
        self._lives = cur_lives

        self._ram = ram
        self._info = info
        self._turn += 1

        truncated = truncated or self._turn >= self.config.max_turns

        if terminated or truncated:
            info = dict(info)
            info["outcome"] = f"score:{int(self._score)}"

        return StepResult(
            observation=self._observation(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def state_hash(self) -> str:
        raw = bytes(self._ram) + str(self._turn).encode()
        return hashlib.sha1(raw).hexdigest()[:16]

    def vector_obs(self) -> list[float]:
        return [b / 255.0 for b in self._ram]

    # ponytail: full undo would need env.unwrapped.clone_state()/restore_state()
    # bridged through pickling; not needed because the space-invaders yaml
    # never enables the backtrack intervention. Raise instead of letting the
    # base class's pickle-__dict__ default silently fail on the env handle.
    def snapshot(self) -> bytes:
        raise NotImplementedError("no backtrack for ALE games")

    def restore(self, snap: bytes) -> None:
        raise NotImplementedError("no backtrack for ALE games")

    def system_prompt(self) -> str:
        raise NotImplementedError(
            "system_prompt() is per-game — implement it in the GymnasiumGameAdapter subclass"
        )

    @classmethod
    def eval_suite(cls):
        raise NotImplementedError(
            "eval_suite() is per-game — implement it in the GymnasiumGameAdapter subclass"
        )

    def _observation(self) -> Observation:
        # env `reward` is authoritative for score (never decode score from
        # RAM) — the running total lives on the adapter, so it is passed to
        # the renderer via `info` rather than decoded from `raw_obs`.
        render_info = dict(self._info)
        render_info["score"] = self._score
        text, legal_actions = self.renderer.render(self._ram, render_info)
        return Observation(
            text=text,
            legal_actions=legal_actions,
            turn=self._turn,
            metadata={
                "score": self._score,
                "lives": self._lives,
                "decision": self._turn,
            },
        )
