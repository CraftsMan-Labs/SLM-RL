"""Minimal loop: the tier-selected SLM plays Mastermind.

Runnable once Phase 1 lands. Works on the any-8gb tier.
"""

from slm_rl.config.loader import load_game_config, load_tiers
from slm_rl.games.registry import get_game
from slm_rl.inference.base import create_backend
from slm_rl.platform.hardware import resolve_tier


def main() -> None:
    tier = resolve_tier(load_tiers())
    print(f"tier: {tier.name} -> {tier.model} on {tier.backend}")

    backend = create_backend(tier.backend, tier.model, tier.quantization)

    from slm_rl.agents.llm_agent import LLMAgent

    game_cls = get_game("mastermind")
    game = game_cls(load_game_config("mastermind"))
    agent = LLMAgent(backend, game.system_prompt())

    obs = game.reset(seed=0)
    history = []
    while True:
        decision = agent.act(obs, history)
        history.append(decision)
        result = game.step(decision.action)
        print(f"guess: {decision.action.id} -> {result.observation.text}")
        if result.terminated or result.truncated:
            print(f"outcome: {result.info.get('outcome')}")
            break
        obs = result.observation


if __name__ == "__main__":
    main()
