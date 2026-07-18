"""EpisodeRunner: drives Agent x Game with the DoomLoopMonitor wired in,
streaming every decision to the RolloutWriter."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from slm_rl.agents.base import ActionDecision, Agent
from slm_rl.config.schema import GameConfig
from slm_rl.datagen.schema import RolloutRecord
from slm_rl.datagen.writer import RolloutWriter
from slm_rl.games.base import Game
from slm_rl.rollout.monitor import DoomLoopMonitor

REFLECT_NUDGE = (
    "You appear to be stuck ({reason}). Do NOT repeat your previous move. "
    "List moves you have not tried yet and pick one of those."
)


@dataclass
class StepOutcome:
    shaped: float
    terminated: bool
    truncated: bool
    done: bool
    outcome: str | None  # set when done (for record + summary)
    monitor_flags: dict
    intervention: Any
    truncated_by_monitor: bool


def finalize_step(
    cfg: GameConfig,
    monitor: DoomLoopMonitor,
    decision: ActionDecision,
    result,
    state_hash: str,
) -> StepOutcome:
    """Shared post-step: monitor, shaped reward, terminal flags, outcome."""
    intervention = monitor.observe(decision, result, state_hash)
    terminated, truncated = result.terminated, result.truncated
    truncated_by_monitor = False
    if intervention and intervention.kind == "truncate":
        truncated = True
        truncated_by_monitor = True

    parse_penalty = {
        "retry_ok": cfg.retry_penalty,
        "fallback_random": cfg.invalid_action_penalty,
    }.get(decision.parse_status, 0.0)

    shaped = result.reward + result.info.get("shaping", 0.0) + parse_penalty
    if intervention:
        shaped += intervention.penalty
    shaped = max(-1.0, min(1.0, shaped))  # reward hygiene: clip

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

    return StepOutcome(
        shaped=shaped,
        terminated=terminated,
        truncated=truncated,
        done=done,
        outcome=outcome if done else None,
        monitor_flags=monitor_flags,
        intervention=intervention,
        truncated_by_monitor=truncated_by_monitor,
    )


def apply_intervention(obs, intervention, action_id: str):
    """Advance obs for reflect/mask_action interventions (shared by runners)."""
    if intervention is None:
        return obs
    if intervention.kind == "reflect":
        obs.metadata["nudge"] = REFLECT_NUDGE.format(reason=intervention.reason)
    elif intervention.kind == "mask_action":
        obs = replace(
            obs,
            legal_actions=[a for a in obs.legal_actions if a.id != action_id]
            or list(obs.legal_actions),
        )
    # backtrack lands in Phase 3 (needs snapshot cadence + re-render);
    # until then an escalation past reflect goes straight to truncate.
    return obs


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
        cum_reward = 0.0
        invalid_steps = 0
        step_idx = 0
        outcome: str | None = None

        while True:
            if self.pruner is not None:
                obs = self.pruner.prune(obs, seed)
            decision = self.agent.act(obs)
            result = self.game.step(decision.action)
            state_hash = self.game.state_hash()

            step = finalize_step(self.cfg, monitor, decision, result, state_hash)
            if decision.parse_status == "fallback_random":
                invalid_steps += 1
            cum_reward += step.shaped
            outcome = step.outcome

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
                        shaped_reward=step.shaped,
                        cum_reward=cum_reward,
                        terminated=step.terminated,
                        truncated=step.truncated,
                        outcome=step.outcome,
                        state_hash=state_hash,
                        monitor_flags=step.monitor_flags,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                )

            if step.done:
                break

            obs = apply_intervention(result.observation, step.intervention, decision.action.id)
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
