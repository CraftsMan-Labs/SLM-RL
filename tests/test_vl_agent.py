"""VLAgent: message shape (system + [text, image] parts), parser/retry
reuse (imported from llm_agent, not copied), and the recorded
prompt_messages placeholder-not-raw-bytes contract (plan 011 hard rule 3).
No real model/backend -- a scripted FakeBackend, same pattern as
tests/test_generation.py."""

from __future__ import annotations

import hashlib

import pytest

pytest.importorskip("PIL")  # VLAgent._load_image opens frames via Pillow

from slm_rl.agents.vl_agent import VLAgent, build_vl_messages
from slm_rl.games.base import ActionSpec, Observation
from slm_rl.inference.base import GenOutput, GenParams, InferenceBackend
from slm_rl.webui.png import encode_rgb

LEGAL = [
    ActionSpec(id="NOOP", label="stand still"),
    ActionSpec(id="FIRE", label="punch"),
    ActionSpec(id="RIGHT", label="move right"),
    ActionSpec(id="LEFT", label="move left"),
]

# A real (tiny, valid) PNG -- VLAgent._load_image runs it through PIL.Image.open,
# so a bare magic-byte stub (not a real PNG) would raise inside PIL.
FRAME_PNG = encode_rgb(bytes([10, 20, 30]) * (2 * 2), width=2, height=2)


def make_obs(**metadata_overrides):
    metadata = {"frame_png": FRAME_PNG}
    metadata.update(metadata_overrides)
    return Observation(
        text="You are the right paddle. Your score: 0.",
        legal_actions=LEGAL,
        turn=1,
        metadata=metadata,
    )


class ScriptedBackend(InferenceBackend):
    """Returns queued completions in order, one per generate() call."""

    def __init__(self, completions: list[str]):
        self.completions = list(completions)
        self.calls: list[list[dict]] = []

    def generate(self, chats, params: GenParams):
        self.calls.append(chats[0])
        text = self.completions.pop(0)
        return [GenOutput(text=text)]


def test_build_vl_messages_shape_system_text_and_user_image_text():
    image = object()
    messages = build_vl_messages("sys prompt", make_obs(), image)
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == [{"type": "text", "text": "sys prompt"}]

    assert messages[1]["role"] == "user"
    parts = messages[1]["content"]
    assert parts[0] == {"type": "image", "image": image}
    assert parts[1]["type"] == "text"
    assert "ACTION:" in parts[1]["text"]


def test_act_ok_first_try_records_placeholder_not_raw_bytes():
    backend = ScriptedBackend(["ACTION: RIGHT"])
    agent = VLAgent(backend, "sys prompt", seed=0)
    decision = agent.act(make_obs())

    assert decision.action.id == "RIGHT"
    assert decision.parse_status == "ok"

    # The recorded prompt_messages must never carry raw image bytes/PIL
    # objects (plan 011 hard rule 3) -- only a placeholder + sha1.
    user_parts = decision.prompt_messages[1]["content"]
    image_part = next(p for p in user_parts if p["type"] == "image_ref")
    assert "image" not in image_part  # no raw bytes/object leaked through
    assert image_part["sha1"] == hashlib.sha1(FRAME_PNG).hexdigest()
    assert image_part["note"]


def test_act_reuses_llm_agent_parser_retry_ladder():
    # A bad completion (no ACTION: line, no legal id mentioned) triggers the
    # same retry LLMAgent.act would (llm_agent.parse_action returns None ->
    # a retry turn is appended and a second generate() call is made).
    backend = ScriptedBackend(["garbage no move here", "ACTION: LEFT"])
    agent = VLAgent(backend, "sys prompt", seed=0)
    decision = agent.act(make_obs())

    assert len(backend.calls) == 2  # first try + retry
    assert decision.action.id == "LEFT"
    assert decision.parse_status == "retry_ok"
    # retry turn appended: assistant's bad completion + a corrective user msg
    retry_call = backend.calls[1]
    assert retry_call[-2]["role"] == "assistant"
    assert retry_call[-1]["role"] == "user"


def test_act_falls_back_to_random_legal_move_after_failed_retry():
    backend = ScriptedBackend(["nonsense", "still nonsense"])
    agent = VLAgent(backend, "sys prompt", seed=42)
    decision = agent.act(make_obs())

    assert decision.parse_status == "fallback_random"
    assert decision.action in LEGAL


def test_act_includes_nudge_when_present():
    backend = ScriptedBackend(["ACTION: NOOP"])
    agent = VLAgent(backend, "sys prompt", seed=0)
    obs = make_obs(nudge="stop repeating yourself")
    agent.act(obs)

    user_text = next(p["text"] for p in backend.calls[0][1]["content"] if p["type"] == "text")
    assert "stop repeating yourself" in user_text


def test_retry_messages_also_redacted_in_record():
    backend = ScriptedBackend(["garbage", "ACTION: FIRE"])
    agent = VLAgent(backend, "sys prompt", seed=0)
    decision = agent.act(make_obs())

    # the retry-path recorded messages include the ORIGINAL user turn's
    # image part, which must also be redacted (not just the first attempt).
    first_user_parts = decision.prompt_messages[1]["content"]
    image_part = next(p for p in first_user_parts if p["type"] == "image_ref")
    assert image_part["sha1"] == hashlib.sha1(FRAME_PNG).hexdigest()
