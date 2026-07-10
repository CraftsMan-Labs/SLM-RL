"""ConsistentMenuPruner: menu contents, determinism, and the runner
integration that makes repeat doom-loops structurally impossible."""

from slm_rl.agents.base import ActionDecision, Agent
from slm_rl.config.loader import load_game_config
from slm_rl.datagen.schema import RolloutRecord
from slm_rl.datagen.writer import RolloutWriter
from slm_rl.games.base import Observation
from slm_rl.games.mastermind.env import MastermindGame
from slm_rl.rollout.runner import EpisodeRunner
from slm_rl.teachers import make_pruner

CFG = load_game_config("mastermind")


def obs_with(history, turn=0, **extra_meta):
    return Observation(
        text="x", legal_actions=[], turn=turn, metadata={"history": history, **extra_meta}
    )


def test_prune_caps_and_sorts():
    pruner = make_pruner(CFG, top_k=10)
    pruned = pruner.prune(obs_with([]), episode_seed=7)
    ids = [a.id for a in pruned.legal_actions]
    assert len(ids) == 10
    assert ids == sorted(ids)


def test_prune_full_set_when_small_and_secret_present():
    pruner = make_pruner(CFG, top_k=10)
    # history pins the candidate set to 4 codes (see test_solver golden)
    hist = [["RGBO", 3, 0], ["RGBP", 3, 0]]
    pruned = pruner.prune(obs_with(hist), episode_seed=7)
    ids = [a.id for a in pruned.legal_actions]
    assert ids == ["RGBB", "RGBG", "RGBR", "RGBY"]  # full consistent set, sorted
    assert not {g for g, _, _ in hist} & set(ids)  # no played guess, ever


def test_prune_deterministic_per_seed_and_turn():
    pruner = make_pruner(CFG, top_k=10)
    a = [x.id for x in pruner.prune(obs_with([], turn=0), 7).legal_actions]
    b = [x.id for x in pruner.prune(obs_with([], turn=0), 7).legal_actions]
    c = [x.id for x in pruner.prune(obs_with([], turn=1), 7).legal_actions]
    d = [x.id for x in pruner.prune(obs_with([], turn=0), 8).legal_actions]
    assert a == b
    assert a != c and a != d


def test_prune_noop_without_history_and_preserves_metadata():
    pruner = make_pruner(CFG)
    plain = Observation(text="x", legal_actions=[], turn=0)  # no history key
    assert pruner.prune(plain, 1) is plain
    nudged = pruner.prune(obs_with([], nudge="stop looping"), 1)
    assert nudged.metadata["nudge"] == "stop looping"


class FirstItemAgent(Agent):
    """DegenerateAgent's twin: always plays menu slot 1. Without the pruner
    this doom-loops; with it, repeats are impossible."""

    def act(self, obs, history) -> ActionDecision:
        action = obs.legal_actions[0]
        return ActionDecision(action=action, raw_completion=f"ACTION: {action.id}")


def test_pruned_runner_kills_repeat_loops(tmp_path):
    out = tmp_path / "pruned.jsonl"
    with RolloutWriter(out) as writer:
        for i in range(5):
            runner = EpisodeRunner(
                MastermindGame(CFG), FirstItemAgent(), CFG, writer=writer,
                pruner=make_pruner(CFG, top_k=10),
            )
            summary = runner.run_episode(seed=300 + i, episode_id=f"p-{i}")
            # the same agent without a pruner triggers reflect->truncate
            # (test_monitor_runner); pruned menus never repeat a guess
            assert summary["monitor"]["interventions"] == 0
            assert summary["outcome"] in ("win", "loss")

    records = [RolloutRecord.from_json(l) for l in out.read_text().splitlines()]
    assert all(len(r.legal_actions) <= 10 for r in records)  # records see the pruned menu
    for ep in {r.episode_id for r in records}:
        played = [r.parsed_action for r in records if r.episode_id == ep]
        assert len(played) == len(set(played))  # no repeats, structurally
