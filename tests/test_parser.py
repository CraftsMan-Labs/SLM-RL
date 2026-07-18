"""Golden tests for the D3 parsing ladder."""

from slm_rl.agents.base import ActionDecision
from slm_rl.agents.llm_agent import LLMAgent, parse_action
from slm_rl.games.base import ActionSpec, Observation
from slm_rl.inference.base import GenOutput, GenParams, InferenceBackend

LEGAL = [ActionSpec(id=c, label=c) for c in ("r0c0", "r0c1", "r1c0", "r1c1")]


def test_exact_action_line():
    assert parse_action("I'll try the corner.\nACTION: r0c1", LEGAL).id == "r0c1"


def test_index_answer():
    assert parse_action("ACTION: 3", LEGAL).id == "r1c0"


def test_case_insensitive_and_decorated():
    assert parse_action("action: **r0c0**.", LEGAL).id == "r0c0"


def test_last_action_line_wins():
    text = "ACTION: r0c0\nwait no...\nACTION: r1c1"
    assert parse_action(text, LEGAL).id == "r1c1"


def test_mention_without_action_line():
    assert parse_action("I think r1c0 is best given the feedback", LEGAL).id == "r1c0"


def test_garbage_returns_none():
    assert parse_action("I have no idea what to do!!!", LEGAL) is None


def test_out_of_range_index_falls_through():
    # index 9 invalid, but r1c1 appears in the reasoning
    assert parse_action("Options considered: r1c1\nACTION: 9", LEGAL).id == "r1c1"


class ScriptedBackend(InferenceBackend):
    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.calls: list[list[dict]] = []

    def generate(self, chats, params: GenParams):
        self.calls.append(chats[0])
        return [GenOutput(text=self.outputs.pop(0))]


def obs() -> Observation:
    return Observation(text="Turn 0. All cells hidden.", legal_actions=LEGAL, turn=0)


def agent(backend) -> LLMAgent:
    return LLMAgent(backend, system_prompt="play boxing", seed=0)


def test_agent_ok_first_try():
    backend = ScriptedBackend(["thinking... ACTION: r0c0"])
    decision = agent(backend).act(obs())
    assert decision.parse_status == "ok"
    assert decision.action.id == "r0c0"
    assert len(backend.calls) == 1


def test_agent_retry_recovers():
    backend = ScriptedBackend(["no clue!!!", "sorry. ACTION: 2"])
    decision = agent(backend).act(obs())
    assert decision.parse_status == "retry_ok"
    assert decision.action.id == "r0c1"
    # retry prompt contains the error feedback
    assert "not a valid move" in backend.calls[1][-1]["content"]


def test_agent_fallback_random_after_two_failures():
    backend = ScriptedBackend(["???", "???"])
    decision = agent(backend).act(obs())
    assert decision.parse_status == "fallback_random"
    assert decision.action in LEGAL


def test_nudge_is_injected_into_prompt():
    backend = ScriptedBackend(["ACTION: r0c0"])
    o = obs()
    o.metadata["nudge"] = "Do NOT repeat your previous move."
    agent(backend).act(o)
    assert "Do NOT repeat" in backend.calls[0][-1]["content"]


def test_build_messages_module_function_matches_agent():
    # teachers stamp prompts via build_messages; it must render exactly what
    # an LLMAgent sends (menu <= 30 actions, format instruction above)
    from slm_rl.agents.llm_agent import build_messages

    small = build_messages("sys", obs())
    assert small[0] == {"role": "system", "content": "sys"}
    assert "1) r0c0" in small[1]["content"]
    assert small[1]["content"].endswith("ACTION: <your move>")

    big_legal = [ActionSpec(id=f"A{i}", label=f"A{i}") for i in range(31)]
    big = build_messages(
        "sys",
        Observation(text="t", legal_actions=big_legal, turn=0,
                    metadata={"action_format": "rRcC e.g. ACTION: r0c0"}),
    )
    assert "rRcC e.g. ACTION: r0c0" in big[1]["content"]
    assert "1)" not in big[1]["content"]

    backend = ScriptedBackend(["ACTION: r0c0"])
    agent(backend).act(obs())
    assert backend.calls[0][1]["content"] == build_messages("play boxing", obs())[1]["content"]
