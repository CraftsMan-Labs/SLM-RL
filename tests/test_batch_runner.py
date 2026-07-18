"""BatchedEpisodeRunner equivalence tests (plan 005). The serial-equivalence
test is the done-or-not-done test: K games batched through one generate()
call per turn must produce identical per-episode outcome/steps/cum_reward
and record parsed_action sequences to the serial EpisodeRunner, in input
order.

Uses StubRepeatGame (fixed menu, win on A) so scripts are trivial and no
ALE / keeper game is required. Tags in observation text route batched
chats to the right script.
"""

from __future__ import annotations

from slm_rl.config.schema import GameConfig, MonitorConfig
from slm_rl.datagen.schema import RolloutRecord
from slm_rl.games.base import ActionSpec, Game, Observation, StepResult
from slm_rl.inference.base import GenOutput, GenParams, InferenceBackend
from slm_rl.rollout.batch_runner import BatchedEpisodeRunner
from slm_rl.rollout.runner import EpisodeRunner


class StubRepeatGame(Game):
    """Fixed 4-action menu; ACTION: A wins."""

    name = "stub-repeat"

    def __init__(self, config, tag: str | None = None, opponent=None):
        super().__init__(config, opponent)
        self.tag = tag
        self._turn = 0

    def _obs(self) -> Observation:
        text = "pick a letter"
        if self.tag:
            text = f"[[{self.tag}]] {text}"
        return Observation(
            text=text,
            legal_actions=[ActionSpec(id=c, label=c) for c in ("A", "B", "C", "D")],
            turn=self._turn,
        )

    def reset(self, seed=None):
        self._turn = 0
        return self._obs()

    def step(self, action: ActionSpec) -> StepResult:
        self._turn += 1
        win = action.id == "A"
        truncated = (not win) and self._turn >= self.config.max_turns
        return StepResult(
            observation=self._obs(),
            reward=1.0 if win else 0.0,
            terminated=win,
            truncated=truncated,
            info={"outcome": "win" if win else ("loss" if truncated else None)},
        )

    def state_hash(self) -> str:
        return f"t{self._turn}"

    def system_prompt(self) -> str:
        return "play stub"

    @classmethod
    def eval_suite(cls):
        from slm_rl.eval.suites import EvalSuite

        return EvalSuite(game="stub-repeat", seeds=(0,), primary_metric="win_rate")


def stub_cfg(*, action_repeat_threshold: int = 99, **overrides) -> GameConfig:
    return GameConfig(
        name="stub-repeat",
        max_turns=6,
        monitor=MonitorConfig(
            interventions=["reflect", "truncate"],
            action_repeat_threshold=action_repeat_threshold,
        ),
        **overrides,
    )


class ScriptedBatchBackend(InferenceBackend):
    """Routes each chat in a batched generate() call to the script whose tag
    appears in the chat's user message, and pops that script's next
    completion. Batch order/size never matters -- only the tag does."""

    def __init__(self, scripts_by_tag: dict[str, list[str]]):
        self.scripts = {k: list(v) for k, v in scripts_by_tag.items()}
        self.batch_sizes: list[int] = []
        self.batches: list[list[list[dict]]] = []

    def _tag_of(self, chat: list[dict]) -> str:
        full_text = " ".join(m["content"] for m in chat)
        for tag in self.scripts:
            if f"[[{tag}]]" in full_text:
                return tag
        raise AssertionError(f"no script tag matched chat: {full_text!r}")

    def generate(self, chats, params: GenParams):
        self.batch_sizes.append(len(chats))
        self.batches.append([list(c) for c in chats])
        return [GenOutput(text=self.scripts[self._tag_of(chat)].pop(0)) for chat in chats]

    def load_adapter(self, path):
        pass

    def close(self):
        pass


class _CollectingWriter:
    def __init__(self):
        self.records: list[RolloutRecord] = []

    def write(self, record):
        self.records.append(record)


def run_serial(seeds, cfg, scripts_by_tag) -> tuple[list[dict], list[RolloutRecord]]:
    from slm_rl.agents.llm_agent import LLMAgent

    summaries, records = [], []
    for idx, seed in enumerate(seeds):
        tag = f"ep{idx}"
        backend = ScriptedBatchBackend({tag: list(scripts_by_tag[tag])})
        agent = LLMAgent(backend, "play", gen_params=GenParams(), seed=seed)
        writer = _CollectingWriter()
        runner = EpisodeRunner(
            StubRepeatGame(cfg, tag=tag), agent, cfg, writer=writer,
            run_id="serial", generation=0, model_id="fake",
        )
        summaries.append(runner.run_episode(seed, episode_id=f"ep-{idx}"))
        records.extend(writer.records)
    return summaries, records


def run_batched(seeds, cfg, scripts_by_tag):
    games = [StubRepeatGame(cfg, tag=f"ep{idx}") for idx in range(len(seeds))]
    backend = ScriptedBatchBackend(scripts_by_tag)
    writer = _CollectingWriter()
    runner = BatchedEpisodeRunner(
        games=games,
        seeds=list(seeds),
        episode_ids=[f"ep-{i}" for i in range(len(seeds))],
        game_cfg=cfg,
        backend=backend,
        system_prompt="play",
        gen_params=GenParams(),
        writer=writer,
        run_id="batched",
        generation=0,
        model_id="fake",
    )
    summaries = runner.run()
    return summaries, writer.records, backend


def test_serial_equivalence_six_seeds():
    seeds = [0, 1, 2, 3, 4, 5]
    cfg = stub_cfg()
    scripts_by_tag = {f"ep{i}": ["ACTION: A"] for i in range(len(seeds))}

    serial_summaries, serial_records = run_serial(seeds, cfg, scripts_by_tag)
    batched_summaries, batched_records, _ = run_batched(
        seeds, cfg, {k: list(v) for k, v in scripts_by_tag.items()}
    )

    assert len(batched_summaries) == len(serial_summaries) == len(seeds)
    for s_serial, s_batch in zip(serial_summaries, batched_summaries):
        assert s_batch["outcome"] == s_serial["outcome"]
        assert s_batch["steps"] == s_serial["steps"]
        assert s_batch["cum_reward"] == s_serial["cum_reward"]
        assert s_batch["invalid_steps"] == s_serial["invalid_steps"]

    def parsed_sequence(records, episode_id):
        return [r.parsed_action for r in records if r.episode_id == episode_id]

    for i in range(len(seeds)):
        eid = f"ep-{i}"
        assert parsed_sequence(batched_records, eid) == parsed_sequence(serial_records, eid)

    assert all(s["outcome"] == "win" for s in serial_summaries)


def test_ragged_termination_no_finished_episode_leaks_into_later_batches():
    seeds = [0, 1, 4]
    cfg = stub_cfg()
    scripts_by_tag = {
        "ep0": ["ACTION: A"],
        "ep1": ["ACTION: B", "ACTION: C", "ACTION: A"],
        "ep2": ["ACTION: B", "ACTION: C", "ACTION: D", "ACTION: A"],
    }
    summaries, records, backend = run_batched(seeds, cfg, scripts_by_tag)

    assert summaries[0]["steps"] == 1
    assert summaries[0]["outcome"] == "win"
    assert all(s["outcome"] == "win" for s in summaries)

    for batch in backend.batches[1:]:
        for chat in batch:
            assert "[[ep0]]" not in " ".join(m["content"] for m in chat)

    assert backend.batch_sizes[0] == 3
    assert backend.batch_sizes[-1] <= 2

    terminal = [r for r in records if r.terminated or r.truncated]
    assert len(terminal) == 3


def test_retry_isolation_only_garbled_episode_retries():
    seeds = [0, 1, 2]
    cfg = stub_cfg()
    scripts_by_tag = {
        "ep0": ["ACTION: A"],
        # no A/B/C/D letters — last-resort parser would otherwise "succeed"
        "ep1": ["@@@!!!@@@", "ACTION: A"],
        "ep2": ["ACTION: A"],
    }
    summaries, records, backend = run_batched(seeds, cfg, scripts_by_tag)

    assert all(s["outcome"] == "win" for s in summaries)

    def status_sequence(records, episode_id):
        return [r.parse_status for r in records if r.episode_id == episode_id]

    assert "retry_ok" in status_sequence(records, "ep-1")
    assert status_sequence(records, "ep-0") == ["ok"] * summaries[0]["steps"]
    assert status_sequence(records, "ep-2") == ["ok"] * summaries[2]["steps"]

    def chat_text(chat):
        return " ".join(m["content"] for m in chat)

    retry_batches = [
        b for b in backend.batches
        if len(b) == 1 and "[[ep1]]" in chat_text(b[0]) and "not a valid move" in chat_text(b[0])
    ]
    assert len(retry_batches) == 1


def test_monitor_isolation_only_degenerate_episode_triggers_interventions():
    seeds = [0, 1, 2]
    cfg = stub_cfg(action_repeat_threshold=2)
    scripts_by_tag = {
        "ep0": ["ACTION: A"],
        "ep1": ["ACTION: B"] * 6,
        "ep2": ["ACTION: A"],
    }
    summaries, records, backend = run_batched(seeds, cfg, scripts_by_tag)

    assert summaries[0]["monitor"]["interventions"] == 0
    assert summaries[2]["monitor"]["interventions"] == 0

    ep1_kinds = summaries[1]["monitor"]["intervention_kinds"]
    assert ep1_kinds[0] == "reflect"
    assert ep1_kinds[-1] == "truncate"
    assert summaries[1]["outcome"] == "truncated"

    assert summaries[0]["outcome"] == "win"
    assert summaries[2]["outcome"] == "win"
