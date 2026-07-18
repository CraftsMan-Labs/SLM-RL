"""BatchedEpisodeRunner: drives K games in lockstep so a single
`backend.generate(...)` call serves every still-live episode's turn instead
of one call per episode per turn (plan 005).

Post-step reward/monitor/intervention logic lives in
`slm_rl.rollout.runner.finalize_step` / `apply_intervention` (shared with
the serial EpisodeRunner).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone

from slm_rl.agents.base import ActionDecision
from slm_rl.agents.llm_agent import action_instruction, build_messages, parse_action
from slm_rl.config.schema import GameConfig
from slm_rl.datagen.schema import RolloutRecord
from slm_rl.datagen.writer import RolloutWriter
from slm_rl.games.base import Game
from slm_rl.inference.base import GenParams, InferenceBackend
from slm_rl.rollout.monitor import DoomLoopMonitor
from slm_rl.rollout.runner import apply_intervention, finalize_step


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

            actions: dict[int, tuple] = {}  # id(ep) -> (action, text, messages, parse_status)
            retry_eps = []
            for ep, out in zip(live, outputs):
                messages = messages_by_ep[id(ep)]
                action = parse_action(out.text, ep.obs.legal_actions)
                if action is not None:
                    actions[id(ep)] = (action, out.text, messages, "ok")
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
                        actions[id(ep)] = (action, out2.text, retry_messages, "retry_ok")
                    else:
                        fallback = ep.rng.choice(list(ep.obs.legal_actions))
                        actions[id(ep)] = (fallback, out2.text, retry_messages, "fallback_random")

            for ep in live:
                action, completion, prompt_messages, parse_status = actions[id(ep)]
                self._step_episode(ep, action, completion, prompt_messages, parse_status)

        return [self._summary(ep) for ep in episodes]

    def _step_episode(
        self,
        ep: _EpisodeState,
        action,
        completion: str,
        prompt_messages: list[dict],
        parse_status: str,
    ) -> None:
        result = ep.game.step(action)
        state_hash = ep.game.state_hash()
        decision = ActionDecision(action=action, raw_completion=completion, parse_status=parse_status)
        step = finalize_step(self.game_cfg, ep.monitor, decision, result, state_hash)

        if parse_status == "fallback_random":
            ep.invalid_steps += 1
        ep.cum_reward += step.shaped
        ep.outcome = step.outcome

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
                    parsed_action=action.id,
                    legal_actions=[a.id for a in ep.obs.legal_actions],
                    parse_status=parse_status,
                    reward=result.reward,
                    shaped_reward=step.shaped,
                    cum_reward=ep.cum_reward,
                    terminated=step.terminated,
                    truncated=step.truncated,
                    outcome=step.outcome,
                    state_hash=state_hash,
                    monitor_flags=step.monitor_flags,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )

        if step.done:
            ep.done = True
            return

        ep.obs = apply_intervention(result.observation, step.intervention, action.id)
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
