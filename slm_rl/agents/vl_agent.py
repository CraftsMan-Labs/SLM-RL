"""VLAgent: a vision-language player for pixel-observation games (plan 011).
Builds multimodal messages (system + [text, image] parts from
`Observation.metadata["frame_png"]`) and reuses `LLMAgent`'s parser/retry
ladder verbatim (imported, never copied -- `llm_agent.action_instruction`
and `llm_agent.parse_action` are the single source of truth for both).

Recorded `prompt_messages` never carry raw image bytes or a PIL object
(neither is JSONlines-serializable, and the plan-011 hard rule is that
vision records must not poison text SFT): the image part is replaced with
a small `{"type": "image_ref", "note": ..., "sha1": ...}` placeholder.
Games that emit these records must set `Game.export_exempt() -> True` so
`datagen/sft_export.py` skips them. The live viewer's replay
(`webui/replay.py`) regenerates visuals from recorded actions.
"""

from __future__ import annotations

import hashlib
import random

from slm_rl.agents.base import ActionDecision, Agent
from slm_rl.agents.llm_agent import action_instruction, generate_with_retry
from slm_rl.games.base import Observation
from slm_rl.inference.base import GenParams, InferenceBackend

_IMAGE_REF_NOTE = "frame omitted from record"


def _image_part_placeholder(frame_png: bytes) -> dict:
    return {
        "type": "image_ref",
        "note": _IMAGE_REF_NOTE,
        "sha1": hashlib.sha1(frame_png).hexdigest(),
    }


def build_vl_messages(system_prompt: str, obs: Observation, image) -> list[dict]:
    """The exact chat messages a VLAgent sends to the backend. `image` is
    whatever the backend's processor expects for an image part (a PIL.Image
    for VLTransformersBackend, per the LFM2.5-VL model card) -- this module
    stays backend-agnostic and just slots it into the content-parts shape."""
    user_text = obs.text + "\n\n" + action_instruction(obs)
    if nudge := obs.metadata.get("nudge"):
        user_text += f"\n\nIMPORTANT: {nudge}"
    user_text += "\nThink briefly if needed, then end with one line: ACTION: <your move>"
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def _redact_for_record(messages: list[dict], frame_png: bytes) -> list[dict]:
    """Replace every image part with a sha1 placeholder of `frame_png`."""
    redacted = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, list):
            content = [
                _image_part_placeholder(frame_png) if part.get("type") == "image" else part
                for part in content
            ]
        redacted.append({"role": msg["role"], "content": content})
    return redacted


class VLAgent(Agent):
    def __init__(
        self,
        backend: InferenceBackend,
        system_prompt: str,
        gen_params: GenParams | None = None,
        seed: int | None = None,
    ):
        self.backend = backend
        self.system_prompt = system_prompt
        self.params = gen_params or GenParams()
        self._rng = random.Random(seed)

    def _load_image(self, frame_png: bytes):
        from io import BytesIO

        from PIL import Image

        return Image.open(BytesIO(frame_png)).convert("RGB")

    def act(self, obs: Observation) -> ActionDecision:
        frame_png = obs.metadata["frame_png"]
        messages = build_vl_messages(self.system_prompt, obs, self._load_image(frame_png))
        retry_text = (
            "That was not a valid move. "
            + action_instruction(obs)
            + " Reply with a single line: ACTION: <your move>"
        )
        return generate_with_retry(
            self.backend, messages, self.params, obs.legal_actions, self._rng,
            make_retry=lambda text: [
                {"role": "assistant", "content": [{"type": "text", "text": text}]},
                {"role": "user", "content": [{"type": "text", "text": retry_text}]},
            ],
            record=lambda m: _redact_for_record(m, frame_png),
        )
