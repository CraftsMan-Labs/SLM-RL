"""BatchedEpisodeRunner equivalence tests (plan 005). The serial-equivalence
test is the done-or-not-done test: K games batched through one generate()
call per turn must produce identical per-episode outcome/steps/cum_reward
and record parsed_action sequences to the serial EpisodeRunner, in input
order.

Uses a tiny 2-color/2-length Mastermind (4 legal actions: RR, RG, GR, GG) so
secrets are cheap to hand-script deterministically. `TaggedMastermindGame`
stamps a per-episode tag into the observation text so the fake backend can
route each chat to the right script regardless of batch order/shrinkage --
mirroring how a real batched backend is handed a list of chats with no
episode identity of its own, except here the "identity" is readable from
the rendered prompt text (as it would be from any real distinguishing game
state)."""

from __future__ import annotations

from dataclasses import replace

from slm_rl.config.loader import load_game_config
from slm_rl.datagen.schema import RolloutRecord
from slm_rl.games.mastermind import MastermindGame
from slm_rl.inference.base import GenOutput, GenParams, InferenceBackend
from slm_rl.rollout.batch_runner import BatchedEpisodeRunner
from slm_rl.rollout.runner import EpisodeRunner

# hand-computed via MastermindGame(tiny_cfg())._secret per seed (printed once,
# see the reset() call: rng = random.Random(seed); secret drawn from "RG")
SECRETS = {0: "GG", 1: "RR", 2: "RR", 3: "RR", 4: "RG", 5: "GG"}


def tiny_cfg(**overrides):
    cfg = load_game_config("mastermind")
    extra = {"code_length": 2, "num_colors": 2, "allow_duplicates": True}
    updates = {"extra": extra, "max_turns": 6, **overrides}
    return cfg.model_copy(update=updates)


class TaggedMastermindGame(MastermindGame):
    """Stamps f"[[{tag}]]" into every observation's text so a fake backend
    can route batched chats back to the right per-episode script."""

    def __init__(self, config, tag: str, opponent=None):
        super().__init__(config, opponent)
        self.tag = tag

    def reset(self, seed=None):
        obs = super().reset(seed)
        return replace(obs, text=f"[[{self.tag}]] {obs.text}")

    def step(self, action):
        result = super().step(action)
        obs = replace(result.observation, text=f"[[{self.tag}]] {result.observation.text}")
        return replace(result, observation=obs)


class ScriptedBatchBackend(InferenceBackend):
    """Routes each chat in a batched generate() call to the script whose tag
    appears in the chat's user message, and pops that script's next
    completion. Batch order/size never matters -- only the tag does."""

    def __init__(self, scripts_by_tag: dict[str, list[str]]):
        self.scripts = {k: list(v) for k, v in scripts_by_tag.items()}
        self.batch_sizes: list[int] = []
        self.batches: list[list[list[dict]]] = []

    def _tag_of(self, chat: list[dict]) -> str:
        # the tag lives in the ORIGINAL observation turn; retry chats append
        # extra messages after it, so scan the whole chat, not just the tail.
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


def win_script(secret: str) -> list[str]:
    """Guess every code except `secret` first (fixed order), then win --
    always finds the secret within 4 guesses over the 4-code space."""
    order = ["RR", "RG", "GR", "GG"]
    guesses = [g for g in order if g != secret] + [secret]
    return [f"ACTION: {g}" for g in guesses]


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
            TaggedMastermindGame(cfg, tag), agent, cfg, writer=writer,
            run_id="serial", generation=0, model_id="fake",
        )
        summaries.append(runner.run_episode(seed, episode_id=f"ep-{idx}"))
        records.extend(writer.records)
    return summaries, records


def run_batched(seeds, cfg, scripts_by_tag, pruners=None):
    games = [TaggedMastermindGame(cfg, f"ep{idx}") for idx in range(len(seeds))]
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
        pruners=pruners,
    )
    summaries = runner.run()
    return summaries, writer.records, backend


# --- 1. Serial-equivalence -------------------------------------------------


def test_serial_equivalence_six_seeds():
    seeds = [0, 1, 2, 3, 4, 5]
    cfg = tiny_cfg()
    scripts_by_tag = {f"ep{i}": win_script(SECRETS[seeds[i]]) for i in range(len(seeds))}

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

    # all 6 episodes actually won (sanity: the equivalence isn't vacuous)
    assert all(s["outcome"] == "win" for s in serial_summaries)


# --- 2. Ragged termination ---------------------------------------------------


def test_ragged_termination_no_finished_episode_leaks_into_later_batches():
    # episode 0 wins in 1 guess (seed forces secret == first scripted guess);
    # episodes 1/2 take longer, so the batch must shrink to size 1 for the
    # trailing turns and never include episode 0's prompt again.
    seeds = [0, 1, 4]
    cfg = tiny_cfg()
    fast_win = [f"ACTION: {SECRETS[0]}"]  # wins turn 1
    scripts_by_tag = {
        "ep0": fast_win,
        "ep1": win_script(SECRETS[1]),
        "ep2": win_script(SECRETS[4]),
    }
    summaries, records, backend = run_batched(seeds, cfg, scripts_by_tag)

    assert summaries[0]["steps"] == 1
    assert summaries[0]["outcome"] == "win"
    assert all(s["outcome"] == "win" for s in summaries)

    # after turn 1, episode 0 is done -- no later batch call may contain its tag
    for batch in backend.batches[1:]:
        for chat in batch:
            assert "[[ep0]]" not in " ".join(m["content"] for m in chat)

    # batch sizes shrink monotonically as episodes finish (3 -> ... -> <=2)
    assert backend.batch_sizes[0] == 3
    assert backend.batch_sizes[-1] <= 2

    # every episode wrote a terminal record
    terminal = [r for r in records if r.terminated or r.truncated]
    assert len(terminal) == 3


# --- 3. Retry isolation -------------------------------------------------


def test_retry_isolation_only_garbled_episode_retries():
    # episode 1's first output is garbage (unparseable); episodes 0 and 2
    # must be unaffected -- retry_ok only on episode 1, and the retry batch
    # contains exactly its one chat.
    seeds = [0, 1, 2]
    cfg = tiny_cfg()
    scripts_by_tag = {
        "ep0": win_script(SECRETS[0]),
        "ep1": ["I have no idea what to do!!!", f"ACTION: {SECRETS[1]}", *win_script(SECRETS[1])[1:]],
        "ep2": win_script(SECRETS[2]),
    }
    summaries, records, backend = run_batched(seeds, cfg, scripts_by_tag)

    assert all(s["outcome"] == "win" for s in summaries)

    def status_sequence(records, episode_id):
        return [r.parse_status for r in records if r.episode_id == episode_id]

    assert "retry_ok" in status_sequence(records, "ep-1")
    assert status_sequence(records, "ep-0") == ["ok"] * summaries[0]["steps"]
    assert status_sequence(records, "ep-2") == ["ok"] * summaries[2]["steps"]

    # the batch immediately after the garbled turn's main generate() call
    # must be the retry batch, containing exactly episode 1's chat
    def chat_text(chat):
        return " ".join(m["content"] for m in chat)

    retry_batches = [
        b for b in backend.batches
        if len(b) == 1 and "[[ep1]]" in chat_text(b[0]) and "not a valid move" in chat_text(b[0])
    ]
    assert len(retry_batches) == 1


# --- 4. Monitor isolation -------------------------------------------------


def test_monitor_isolation_only_degenerate_episode_triggers_interventions():
    # episode 1 repeats the same guess every turn (degenerate) -- must
    # trigger reflect then truncate (mastermind: action_repeat_threshold=2,
    # ladder [reflect, truncate]); episodes 0 and 2 play normally and must
    # show zero interventions.
    seeds = [0, 1, 2]
    cfg = tiny_cfg()
    # seed 1's secret is RR (see SECRETS), so repeating GR (never the secret)
    # forces the doom loop instead of an accidental turn-1 win.
    assert SECRETS[1] == "RR"
    degenerate = ["ACTION: GR"] * 6
    scripts_by_tag = {
        "ep0": win_script(SECRETS[0]),
        "ep1": degenerate,
        "ep2": win_script(SECRETS[2]),
    }
    summaries, records, backend = run_batched(seeds, cfg, scripts_by_tag)

    assert summaries[0]["monitor"]["interventions"] == 0
    assert summaries[2]["monitor"]["interventions"] == 0

    ep1_kinds = summaries[1]["monitor"]["intervention_kinds"]
    assert ep1_kinds[0] == "reflect"
    assert ep1_kinds[-1] == "truncate"
    assert summaries[1]["outcome"] == "truncated"

    # episodes 0/2 wins are unaffected by episode 1's doom loop
    assert summaries[0]["outcome"] == "win"
    assert summaries[2]["outcome"] == "win"
