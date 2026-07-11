"""Gymnasium adapter: wraps an external `gymnasium.Env` (ALE with
obs_type="ram") into our `Game` contract via a per-game ObservationRenderer
(RAM -> text). Requires the [atari] extra."""

from __future__ import annotations

import hashlib
import importlib.util
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

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
        # Mnih et al. 2015 DQN eval protocol: random no-op starts. Default 0
        # = disabled = exact pre-existing behavior (CODING_GUIDELINE Sec 2:
        # new knobs default to current behavior unchanged).
        self.noop_start_max: int = int(extra.get("noop_start_max", 0))
        # Workshop playground (plan 013): optional reward-shaping hook, a
        # path to a Python file defining shape_reward(ctx: dict) -> float.
        # Resolved eagerly (missing file -> ValueError now, same doctrine as
        # dqn_checkpoint in make_teacher — never a silent fallback) but
        # loaded lazily on first step() so importing this module never pays
        # for importlib machinery when no hook is set (the common case).
        # None -> the code path in step() below is byte-identical to before
        # this knob existed (CODING_GUIDELINE Sec 2: new knobs default to
        # current behavior unchanged).
        reward_hook = extra.get("reward_hook")
        self._reward_hook_path: Path | None = None
        if reward_hook is not None:
            path = Path(reward_hook)
            if not path.is_file():
                raise ValueError(f"reward_hook not found: {reward_hook!r}")
            self._reward_hook_path = path
        self._shape: Callable[[dict[str, Any]], float] | None = None

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

        if self.noop_start_max > 0 and seed is not None:
            # CODING_GUIDELINE Sec 1.4: derived seed is arithmetic, not a
            # hash/tuple. ALE (repeat_action_probability=0.0) is otherwise
            # seed-invariant at reset (measured 2026-07-11): without this,
            # every eval seed replays the identical initial board. k
            # no-op steps before the first real decision make eval seeds
            # meaningful again (Mnih et al. 2015 DQN eval protocol).
            rng = random.Random(seed * 10_007 + 11)
            k = rng.randint(0, self.noop_start_max)
            if "NOOP" in self._action_ids:
                noop_id = self._action_ids.index("NOOP")
            else:
                noop_id = 0  # ponytail: assumes action 0 is a no-op-like action
                             # when the env doesn't expose "NOOP" explicitly;
                             # true fallback would inspect the action semantics.
            for _ in range(k):
                ram, _reward, terminated, truncated, info = env.step(noop_id)
                # These frames precede the first decision -- never counted
                # into score/turn. If life is lost or the episode ends during
                # no-ops (practically impossible in Space Invaders, but
                # guarded), stop early rather than stepping a dead episode.
                self._ram = ram
                self._info = info
                self._lives = info.get("lives")
                if terminated or truncated:
                    break

        return self._observation()

    def _ensure_shape(self) -> Callable[[dict[str, Any]], float]:
        """Load and cache shape_reward from `self._reward_hook_path` on first
        use. A module without a callable `shape_reward` -> ValueError here,
        not at construction (mirrors the file-existence check, which IS at
        construction time — see __init__)."""
        if self._shape is None:
            spec = importlib.util.spec_from_file_location(
                "slm_rl_playground_reward_hook", self._reward_hook_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            shape = getattr(module, "shape_reward", None)
            if not callable(shape):
                raise ValueError(
                    f"reward_hook {str(self._reward_hook_path)!r} has no callable shape_reward(ctx)"
                )
            self._shape = shape
        return self._shape

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
        lives_lost = prev_lives is not None and cur_lives is not None and cur_lives < prev_lives
        if lives_lost:
            reward += self.life_loss_penalty  # penalty is negative, so add
        self._lives = cur_lives

        self._ram = ram
        self._info = info
        self._turn += 1

        truncated = truncated or self._turn >= self.config.max_turns

        if self._reward_hook_path is not None:
            # Workshop playground (plan 013): the hook wraps exactly the
            # built-in formula's result, computed above. Absent -> this
            # branch never runs, so the reward computation is byte-identical
            # to before this knob existed (proven by test_reward_hook.py).
            # Runs AFTER the turn increment and the max_turns cap so the
            # ctx sees the episode's final truncated flag (a cap-ended
            # episode delivers truncated=True) and turn as the 1-indexed
            # decision count. Monitor-side penalties (retry/fallback/
            # truncate) are applied later, in the rollout runner -- out of
            # the hook's reach by design (they are cross-game concerns,
            # not Atari-specific).
            reward = float(self._ensure_shape()(
                {
                    "raw_points": raw_sum,          # ALE points this decision
                    "default_reward": reward,        # what the built-in formula produced
                    "score": self._score,            # cumulative raw score
                    "lives_lost": lives_lost,        # bool: life lost this decision
                    "lives": cur_lives,
                    "turn": self._turn,              # 1-indexed decision count
                    "terminated": terminated,
                    "truncated": truncated,          # includes the max_turns cap
                    "vector_obs": self.vector_obs(), # 128 floats, RAM/255
                }
            ))

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
        metadata: dict[str, Any] = {
            "score": self._score,
            "lives": self._lives,
            "decision": self._turn,
            # RAM-based teachers/pruners (dqn.py) get the vector without game-
            # object access; a list of 128 floats per decision, dwarfed by
            # prompt_messages -- see plan 012.
            "vector_obs": self.vector_obs(),
        }
        # Optional protocol: a renderer may expose decode(raw_obs) -> dict
        # for non-LLM consumers (the heuristic teacher, plan 009) without
        # widening the ObservationRenderer ABC. Duck-typed via getattr so
        # renderers without it (none yet) are unaffected.
        if decode := getattr(self.renderer, "decode", None):
            metadata["state"] = decode(self._ram)
        return Observation(
            text=text,
            legal_actions=legal_actions,
            turn=self._turn,
            metadata=metadata,
        )
