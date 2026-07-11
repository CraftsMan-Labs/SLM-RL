"""mastermind-easy curriculum entry point (plan 006): registration, config,
solver win rate, and GRPO export all generalize from the 4x6 exemplar."""

from pathlib import Path

from slm_rl.config.loader import load_game_config
from slm_rl.datagen.grpo_export import export_grpo_dataset
from slm_rl.games.mastermind.env import MastermindGame
from slm_rl.games.registry import get_game
from slm_rl.rollout.runner import EpisodeRunner
from slm_rl.teachers import make_teacher

CFG = load_game_config("mastermind-easy")


def test_registered_variant_is_mastermind_subclass():
    assert issubclass(get_game("mastermind-easy"), MastermindGame)


def test_registration_does_not_corrupt_standard_game_name():
    # regression: register_game sets cls.name, so registering the same class
    # object under two names would leave both reporting "mastermind-easy"
    assert get_game("mastermind").name == "mastermind"
    assert get_game("mastermind-easy").name == "mastermind-easy"


def test_easy_config_is_64_codes_over_rgby():
    game = get_game("mastermind-easy")(CFG)
    game.reset(seed=0)
    assert len(game._actions) == 64  # 4 colors ** 3 pegs, duplicates allowed
    assert game.colors == "RGBY"


def test_solver_wins_at_least_98_percent_over_50_episodes():
    agent, _ = make_teacher(CFG, seed=0)
    outcomes = []
    for i in range(50):
        runner = EpisodeRunner(get_game("mastermind-easy")(CFG), agent, CFG)
        summary = runner.run_episode(seed=1000 + i, episode_id=f"easy-{i}")
        outcomes.append(summary["outcome"])
    wins = sum(o == "win" for o in outcomes)
    assert wins / len(outcomes) >= 0.98


def test_export_grpo_dataset_accepts_mastermind_easy(tmp_path: Path):
    import json

    seed = 42
    game = get_game("mastermind-easy")(CFG)
    game.reset(seed)
    secret = game._secret

    records = [
        {
            "episode_id": "e1", "step_idx": 0, "seed": seed, "parsed_action": "RRR",
            "prompt_messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "turn 0"},
            ],
        },
        {
            "episode_id": "e1", "step_idx": 1, "seed": seed, "parsed_action": "GGG",
            "prompt_messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "turn 1"},
            ],
        },
    ]
    dataset_path = tmp_path / "r.jsonl"
    dataset_path.write_text("\n".join(json.dumps(r) for r in records))
    out = tmp_path / "grpo.jsonl"

    n = export_grpo_dataset(dataset_path, out, CFG)
    assert n == 2

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    ctxs = [json.loads(r["game_ctx"]) for r in rows]
    assert all(ctx["secret"] == secret for ctx in ctxs)
