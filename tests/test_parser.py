"""Golden tests for the D3 parsing ladder."""

from slm_rl.agents.base import ActionDecision
from slm_rl.agents.llm_agent import LLMAgent, parse_action
from slm_rl.games.base import ActionSpec, Observation
from slm_rl.inference.base import GenOutput, GenParams, InferenceBackend

LEGAL = [ActionSpec(id=c, label=c) for c in ("RGBY", "RGBB", "YYGG", "OPRG")]


def test_exact_action_line():
    assert parse_action("I'll try blue.\nACTION: RGBB", LEGAL).id == "RGBB"


def test_index_answer():
    assert parse_action("ACTION: 3", LEGAL).id == "YYGG"


def test_case_insensitive_and_decorated():
    assert parse_action("action: **rgby**.", LEGAL).id == "RGBY"


def test_last_action_line_wins():
    text = "ACTION: RGBY\nwait no...\nACTION: OPRG"
    assert parse_action(text, LEGAL).id == "OPRG"


def test_mention_without_action_line():
    assert parse_action("I think YYGG is best given the feedback", LEGAL).id == "YYGG"


def test_garbage_returns_none():
    assert parse_action("I have no idea what to do!!!", LEGAL) is None


def test_out_of_range_index_falls_through():
    # index 9 invalid, but OPRG appears in the reasoning
    assert parse_action("Options considered: OPRG\nACTION: 9", LEGAL).id == "OPRG"


class ScriptedBackend(InferenceBackend):
    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.calls: list[list[dict]] = []

    def generate(self, chats, params: GenParams):
        self.calls.append(chats[0])
        return [GenOutput(text=self.outputs.pop(0))]


def obs() -> Observation:
    return Observation(text="No guesses yet.", legal_actions=LEGAL, turn=0)


def agent(backend) -> LLMAgent:
    return LLMAgent(backend, system_prompt="play mastermind", seed=0)


def test_agent_ok_first_try():
    backend = ScriptedBackend(["thinking... ACTION: RGBY"])
    decision = agent(backend).act(obs(), [])
    assert decision.parse_status == "ok"
    assert decision.action.id == "RGBY"
    assert len(backend.calls) == 1


def test_agent_retry_recovers():
    backend = ScriptedBackend(["no clue!!!", "sorry. ACTION: 2"])
    decision = agent(backend).act(obs(), [])
    assert decision.parse_status == "retry_ok"
    assert decision.action.id == "RGBB"
    # retry prompt contains the error feedback
    assert "not a valid move" in backend.calls[1][-1]["content"]


def test_agent_fallback_random_after_two_failures():
    backend = ScriptedBackend(["???", "???"])
    decision = agent(backend).act(obs(), [])
    assert decision.parse_status == "fallback_random"
    assert decision.action in LEGAL


def test_nudge_is_injected_into_prompt():
    backend = ScriptedBackend(["ACTION: RGBY"])
    o = obs()
    o.metadata["nudge"] = "Do NOT repeat your previous move."
    agent(backend).act(o, [])
    assert "Do NOT repeat" in backend.calls[0][-1]["content"]
