"""Generator node consumes ephemeral turn images + auto-routes to a vision LLM.

The generator reads ``turn_images`` / ``vision_llm`` from the runnable
``config["configurable"]`` (where the entry points put them -- see
test_vision_ephemeral). Behaviour (spec FR-004/FR-005/FR-006):

  * active model already vision-capable  -> attach images, use active model
  * active model blind + vision_llm set  -> route to vision_llm, attach images
  * active model blind + no vision_llm   -> drop images, answer text-only,
                                            append a "can't read images" note
"""
from __future__ import annotations

import asyncio

from opsrag.agent.nodes.generator import generate_node
from opsrag.llms.content import ImagePart, is_multimodal

PNG = b"\x89PNG\r\nfake"


class _LLM:
    def __init__(self, name: str, vision: bool) -> None:
        self._name = name
        self._vision = vision
        self.last_messages = None

    @property
    def model_name(self) -> str:
        return self._name

    async def generate(
        self, messages, system_prompt=None, temperature=0.0, purpose=None
    ):
        self.last_messages = messages

        class _R:
            content = "ok"
            model = self._name
            usage: dict = {}
            latency_ms = 1.0

        return _R()


class _Obs:
    async def log_llm_call(self, **k):  # noqa: ANN003
        return None


def _state() -> dict:
    return {
        "query": "what is this",
        "graded_chunks": [],
        "merged_results": [],
        "retrieved_chunks": [],
        "conversation_history": [],
    }


def test_image_makes_user_content_multimodal_on_vision_model() -> None:
    text_llm = _LLM("claude-sonnet-4-20250514", vision=True)  # already sees
    node = generate_node(text_llm, _Obs())
    cfg = {
        "configurable": {
            "turn_images": [ImagePart(PNG, "image/png")],
            "vision_llm": None,
        }
    }
    asyncio.run(node(_state(), cfg))
    user_content = text_llm.last_messages[-1]["content"]
    assert is_multimodal(user_content)
    assert any(p["type"] == "image" for p in user_content)


def test_routes_to_vision_llm_when_active_model_blind() -> None:
    blind = _LLM("text-bison", vision=False)
    seer = _LLM("gemini-3-flash-preview", vision=True)
    node = generate_node(blind, _Obs())
    cfg = {
        "configurable": {
            "turn_images": [ImagePart(PNG, "image/png")],
            "vision_llm": seer,
        }
    }
    asyncio.run(node(_state(), cfg))
    assert seer.last_messages is not None  # vision LLM was used
    assert is_multimodal(seer.last_messages[-1]["content"])
    assert blind.last_messages is None


def test_no_vision_available_drops_image_and_notes() -> None:
    blind = _LLM("text-bison", vision=False)
    node = generate_node(blind, _Obs())
    cfg = {
        "configurable": {
            "turn_images": [ImagePart(PNG, "image/png")],
            "vision_llm": None,
        }
    }
    out = asyncio.run(node(_state(), cfg))
    assert not is_multimodal(blind.last_messages[-1]["content"])  # text-only
    g = out["generation"].lower()
    assert "can't read images" in g or "cannot read images" in g


def test_no_images_is_byte_identical_text_path() -> None:
    """No images + no config -> plain string content, no note."""
    text_llm = _LLM("text-bison", vision=False)
    node = generate_node(text_llm, _Obs())
    out = asyncio.run(node(_state(), None))
    assert isinstance(text_llm.last_messages[-1]["content"], str)
    assert "can't read images" not in out["generation"].lower()
