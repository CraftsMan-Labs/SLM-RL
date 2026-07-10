"""Menu pruner: the teacher's Q-head seam (HYBRID_RL.md seam 2), exact for
Mastermind. Applied by the EpisodeRunner so records, SFT export, and GRPO
export all see the pruned menu."""

from __future__ import annotations

import random
from dataclasses import replace

from slm_rl.games.base import ActionSpec, Observation
from slm_rl.teachers.mastermind_solver import consistent_candidates


class ConsistentMenuPruner:
    """Replaces the full action set with <= top_k feedback-consistent,
    unplayed codes (played guesses are inconsistent with their own feedback,
    so repeats become structurally impossible).

    Uses only the observation's history — never the secret. When the
    consistent set is larger than top_k the secret may be absent from the
    random sample; that's fine, |C| collapses below top_k within a few
    informative guesses. Deterministic per (episode_seed, turn).
    """

    def __init__(self, colors: str, code_length: int, dup_ok: bool, top_k: int = 10):
        self.colors = colors
        self.code_length = code_length
        self.dup_ok = dup_ok
        self.top_k = top_k

    def prune(self, obs: Observation, episode_seed: int) -> Observation:
        history = obs.metadata.get("history")
        if history is None:
            return obs
        cands = consistent_candidates(self.colors, self.code_length, self.dup_ok, history)
        if len(cands) > self.top_k:
            rng = random.Random(episode_seed * 10_007 + obs.turn)  # deterministic per (seed, turn)
            cands = rng.sample(cands, self.top_k)
        # always sorted: canonical menus dedupe better (grpo_export hashes prompts)
        return replace(obs, legal_actions=[ActionSpec(id=c, label=c) for c in sorted(cands)])
