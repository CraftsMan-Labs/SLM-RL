"""EpisodeRunner: drives Agent x Game with the DoomLoopMonitor wired in,
streaming every decision to the RolloutWriter."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from slm_rl.agents.base import Agent
from slm_rl.config.schema import GameConfig
from slm_rl.datagen.schema import RolloutRecord
from slm_rl.datagen.writer import RolloutWriter
from slm_rl.games.base import Game
from slm_rl.rollout.monitor import DoomLoopMonitor

REFLECT_NUDGE = (
    "You appear to be stuck ({reason}). Do NOT repeat your previous move. "
    "List moves you have not tried yet and pick one of those."
)


class EpisodeRunner:
    def __init__(
        self,
        game: Game,
        agent: Agent,
        game_cfg: GameConfig,
        writer: RolloutWriter | None = None,
        run_id: str = "adhoc",
        generation: int = 0,
        model_id: str = "none",
        adapter_ref: str | None = None,
        opponent_id: str | None = None,
        pruner=None,
    ):
        self.game = game
        self.agent = agent
        self.cfg = game_cfg
        self.writer = writer
        self.run_id = run_id
        self.generation = generation
        self.model_id = model_id
        self.adapter_ref = adapter_ref
        self.opponent_id = opponent_id
        self.pruner = pruner  # teacher menu pruning (HYBRID_RL.md seam 2)

    def run_episode(self, seed: int, episode_id: str) -> dict:
        obs = self.game.reset(seed)
        monitor = DoomLoopMonitor(self.cfg.monitor)
        history: list = []
        cum_reward = 0.0
        invalid_steps = 0
        step_idx = 0
        outcome: str | None = None

        while True:
            if self.pruner is not None:
                obs = self.pruner.prune(obs, seed)
            decision = self.agent.act(obs, history)
            history.append(decision)
            result = self.game.step(decision.action)
            state_hash = self.game.state_hash()

            intervention = monitor.observe(decision, result, state_hash)
            terminated, truncated = result.terminated, result.truncated
            truncated_by_monitor = False
            if intervention and intervention.kind == "truncate":
                truncated = True
                truncated_by_monitor = True

            parse_penalty = {
                "retry_ok": self.cfg.retry_penalty,
                "fallback_random": self.cfg.invalid_action_penalty,
            }.get(decision.parse_status, 0.0)
            if decision.parse_status == "fallback_random":
                invalid_steps += 1

            shaped = result.reward + result.info.get("shaping", 0.0) + parse_penalty
            if intervention:
                shaped += intervention.penalty
            shaped = max(-1.0, min(1.0, shaped))  # reward hygiene: clip
            cum_reward += shaped

            done = terminated or truncated
            outcome = result.info.get("outcome")
            if done and outcome is None:
                outcome = "truncated"

            monitor_flags: dict = {}
            if intervention:
                monitor_flags["intervention"] = {
                    "kind": intervention.kind,
                    "reason": intervention.reason,
                    "penalty": intervention.penalty,
                }
            if truncated_by_monitor:
                monitor_flags["truncated_by_monitor"] = True

            if self.writer:
                self.writer.write(
                    RolloutRecord(
                        run_id=self.run_id,
                        generation=self.generation,
                        game=self.cfg.name,
                        episode_id=episode_id,
                        step_idx=step_idx,
                        seed=seed,
                        model_id=self.model_id,
                        adapter_ref=self.adapter_ref,
                        opponent_id=self.opponent_id,
                        prompt_messages=decision.prompt_messages,
                        completion=decision.raw_completion,
                        parsed_action=decision.action.id,
                        legal_actions=[a.id for a in obs.legal_actions],
                        parse_status=decision.parse_status,
                        reward=result.reward,
                        shaped_reward=shaped,
                        cum_reward=cum_reward,
                        terminated=terminated,
                        truncated=truncated,
                        outcome=outcome if done else None,
                        state_hash=state_hash,
                        monitor_flags=monitor_flags,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                )

            if done:
                break

            obs = result.observation
            if intervention:
                if intervention.kind == "reflect":
                    obs.metadata["nudge"] = REFLECT_NUDGE.format(reason=intervention.reason)
                elif intervention.kind == "mask_action":
                    obs = replace(
                        obs,
                        legal_actions=[
                            a for a in obs.legal_actions if a.id != decision.action.id
                        ]
                        or list(obs.legal_actions),
                    )
                # backtrack lands in Phase 3 (needs snapshot cadence + re-render);
                # until then an escalation past reflect goes straight to truncate.
            step_idx += 1

        stats = monitor.episode_stats()
        return {
            "episode_id": episode_id,
            "seed": seed,
            "outcome": outcome,
            "cum_reward": cum_reward,
            "steps": step_idx + 1,
            "invalid_steps": invalid_steps,
            "monitor": stats,
        }
