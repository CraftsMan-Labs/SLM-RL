"""BatchedEpisodeRunner: drives K games in lockstep so a single
`backend.generate(...)` call serves every still-live episode's turn instead
of one call per episode per turn (plan 005).

# ponytail: the per-step logic below (monitor.observe -> intervention
# handling -> RolloutRecord write -> obs advance) is a deliberate copy of
# slm_rl/rollout/runner.py's EpisodeRunner.run_episode loop body. It is NOT
# extracted into a shared helper -- runner.py is the serial reference
# implementation and must stay untouched (plan 005 scope).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from slm_rl.agents.base import ActionDecision
from slm_rl.agents.llm_agent import action_instruction, build_messages, parse_action
from slm_rl.config.schema import GameConfig
from slm_rl.datagen.schema import RolloutRecord
from slm_rl.datagen.writer import RolloutWriter
from slm_rl.games.base import Game
from slm_rl.inference.base import GenParams, InferenceBackend
from slm_rl.rollout.monitor import DoomLoopMonitor
from slm_rl.rollout.runner import REFLECT_NUDGE


@dataclass
class _EpisodeState:
    """Per-episode mutable state; never shared across episodes."""

    game: Game
    seed: int
    episode_id: str
    pruner: object | None
    monitor: DoomLoopMonitor
    rng: random.Random  # seeded fallback sampling, one per episode (D4 determinism)
    obs: object
    cum_reward: float = 0.0
    invalid_steps: int = 0
    step_idx: int = 0
    outcome: str | None = None
    done: bool = False


@dataclass
class BatchedEpisodeRunner:
    """Constructor mirrors EpisodeRunner but drives `games` (list[Game]) in
    lockstep against a shared `backend`, batching each turn's `generate`
    call across all still-live episodes.

    The Agent-level retry ladder (LLMAgent.act) is bypassed on purpose: this
    runner talks to the backend directly so it can batch across episodes.
    Its retry semantics are equivalent -- one retry generate call per failed
    parse, then the seeded random fallback -- just batched across whichever
    episodes need a retry on a given turn.
    """

    games: list[Game]
    seeds: list[int]
    episode_ids: list[str]
    game_cfg: GameConfig
    backend: InferenceBackend
    system_prompt: str
    gen_params: GenParams
    writer: RolloutWriter | None = None
    run_id: str = "adhoc"
    generation: int = 0
    model_id: str = "none"
    adapter_ref: str | None = None
    opponent_id: str | None = None
    pruners: list[object | None] | None = None  # per-episode pruner, parallel to `games`

    def run(self) -> list[dict]:
        n = len(self.games)
        pruners = self.pruners if self.pruners is not None else [None] * n
        episodes = [
            _EpisodeState(
                game=game,
                seed=seed,
                episode_id=episode_id,
                pruner=pruner,
                monitor=DoomLoopMonitor(self.game_cfg.monitor),
                rng=random.Random(seed),
                obs=game.reset(seed),
            )
            for game, seed, episode_id, pruner in zip(
                self.games, self.seeds, self.episode_ids, pruners
            )
        ]

        while True:
            live = [ep for ep in episodes if not ep.done]
            if not live:
                break
            for ep in live:
                if ep.pruner is not None:
                    ep.obs = ep.pruner.prune(ep.obs, ep.seed)

            messages_by_ep = {id(ep): build_messages(self.system_prompt, ep.obs) for ep in live}
            outputs = self.backend.generate(
                [messages_by_ep[id(ep)] for ep in live], self.gen_params
            )

            actions: dict[int, tuple] = {}  # id(ep) -> (action, text, messages, parse_status, logprob)
            retry_eps = []
            for ep, out in zip(live, outputs):
                messages = messages_by_ep[id(ep)]
                action = parse_action(out.text, ep.obs.legal_actions)
                if action is not None:
                    actions[id(ep)] = (action, out.text, messages, "ok", out.logprob)
                else:
                    retry_eps.append((ep, messages, out.text))

            if retry_eps:
                retry_messages_by_ep = {
                    id(ep): messages
                    + [
                        {"role": "assistant", "content": text},
                        {
                            "role": "user",
                            "content": (
                                "That was not a valid move. "
                                + action_instruction(ep.obs)
                                + " Reply with a single line: ACTION: <your move>"
                            ),
                        },
                    ]
                    for ep, messages, text in retry_eps
                }
                retry_outputs = self.backend.generate(
                    [retry_messages_by_ep[id(ep)] for ep, _, _ in retry_eps], self.gen_params
                )
                for (ep, _messages, _text), out2 in zip(retry_eps, retry_outputs):
                    retry_messages = retry_messages_by_ep[id(ep)]
                    action = parse_action(out2.text, ep.obs.legal_actions)
                    if action is not None:
                        actions[id(ep)] = (action, out2.text, retry_messages, "retry_ok", out2.logprob)
                    else:
                        fallback = ep.rng.choice(list(ep.obs.legal_actions))
                        actions[id(ep)] = (fallback, out2.text, retry_messages, "fallback_random", None)

            for ep in live:
                action, completion, prompt_messages, parse_status, logprob = actions[id(ep)]
                self._step_episode(ep, action, completion, prompt_messages, parse_status, logprob)

        return [self._summary(ep) for ep in episodes]

    def _step_episode(
        self,
        ep: _EpisodeState,
        action,
        completion: str,
        prompt_messages: list[dict],
        parse_status: str,
        logprob: float | None,
    ) -> None:
        result = ep.game.step(action)
        state_hash = ep.game.state_hash()

        decision_action_id = action.id
        # DoomLoopMonitor.observe only reads decision.action.id and
        # decision.parse_status; raw_completion/prompt_messages are unused here.
        decision = ActionDecision(action=action, raw_completion=completion, parse_status=parse_status)
        intervention = ep.monitor.observe(decision, result, state_hash)
        terminated, truncated = result.terminated, result.truncated
        truncated_by_monitor = False
        if intervention and intervention.kind == "truncate":
            truncated = True
            truncated_by_monitor = True

        parse_penalty = {
            "retry_ok": self.game_cfg.retry_penalty,
            "fallback_random": self.game_cfg.invalid_action_penalty,
        }.get(parse_status, 0.0)
        if parse_status == "fallback_random":
            ep.invalid_steps += 1

        shaped = result.reward + result.info.get("shaping", 0.0) + parse_penalty
        if intervention:
            shaped += intervention.penalty
        shaped = max(-1.0, min(1.0, shaped))  # reward hygiene: clip
        ep.cum_reward += shaped

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
                    game=self.game_cfg.name,
                    episode_id=ep.episode_id,
                    step_idx=ep.step_idx,
                    seed=ep.seed,
                    model_id=self.model_id,
                    adapter_ref=self.adapter_ref,
                    opponent_id=self.opponent_id,
                    prompt_messages=prompt_messages,
                    completion=completion,
                    parsed_action=decision_action_id,
                    legal_actions=[a.id for a in ep.obs.legal_actions],
                    parse_status=parse_status,
                    reward=result.reward,
                    shaped_reward=shaped,
                    cum_reward=ep.cum_reward,
                    terminated=terminated,
                    truncated=truncated,
                    outcome=outcome if done else None,
                    state_hash=state_hash,
                    monitor_flags=monitor_flags,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )

        ep.outcome = outcome
        if done:
            ep.done = True
            return

        ep.obs = result.observation
        if intervention:
            if intervention.kind == "reflect":
                ep.obs.metadata["nudge"] = REFLECT_NUDGE.format(reason=intervention.reason)
            elif intervention.kind == "mask_action":
                ep.obs = replace(
                    ep.obs,
                    legal_actions=[
                        a for a in ep.obs.legal_actions if a.id != decision_action_id
                    ]
                    or list(ep.obs.legal_actions),
                )
            # backtrack lands in Phase 3 (needs snapshot cadence + re-render);
            # until then an escalation past reflect goes straight to truncate.
        ep.step_idx += 1

    def _summary(self, ep: _EpisodeState) -> dict:
        stats = ep.monitor.episode_stats()
        return {
            "episode_id": ep.episode_id,
            "seed": ep.seed,
            "outcome": ep.outcome,
            "cum_reward": ep.cum_reward,
            "steps": ep.step_idx + 1,
            "invalid_steps": ep.invalid_steps,
            "monitor": stats,
        }
