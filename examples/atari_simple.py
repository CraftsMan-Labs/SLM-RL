"""Canonical hello-world: random agent plays Atari Freeway.

This recreates the shape of the OpenEnv reference example
(huggingface/OpenEnv examples/atari_simple.py) against our own API —
same idea (reset -> sample from legal_actions -> step -> reward/done),
but in-process with no Docker so it runs on an 8GB machine.

Runnable once Phase 3 (ALE adapter) lands. Requires: pip install "slm-rl[atari]"
"""

import random

from slm_rl.config.loader import load_game_config
from slm_rl.games.registry import get_game


def main() -> None:
    game_cls = get_game("atari_freeway")
    game = game_cls(load_game_config("atari_freeway"))

    obs = game.reset(seed=42)
    total_reward, done = 0.0, False

    for _ in range(100):
        if done:
            break
        action = random.choice(obs.legal_actions)  # mirrors the OpenEnv sample
        result = game.step(action)
        obs, done = result.observation, result.terminated or result.truncated
        total_reward += result.reward
        print(f"turn={obs.turn} action={action.id} reward={result.reward}")

    print(f"episode done, total reward: {total_reward}")


if __name__ == "__main__":
    main()
