"""Unit tests for `_assemble_gemini_contents` ordering / system-prompt placement.

The Gemini path in `opsrag/llms/vertex.py` has no separate system role on
this SDK path, so the system prompt must ride on a USER-role turn -- never
a `model` turn. These tests use lightweight fakes for `Content`/`Part` so
they run without the `vertexai` SDK installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from opsrag.llms.vertex import _assemble_gemini_contents


@dataclass
class _FakePart:
    text: str


@dataclass
class _FakeContent:
    role: str
    parts: list = field(default_factory=list)


def _part_from_text(text: str) -> _FakePart:
    return _FakePart(text=text)


def _parts_builder(content):
    # Mirror to_gemini_parts' string case: one text part.
    return [_FakePart(text=str(content))]


def _assemble(messages, system_prompt):
    return _assemble_gemini_contents(
        messages,
        system_prompt,
        content_cls=_FakeContent,
        part_from_text=_part_from_text,
        parts_builder=_parts_builder,
    )


def test_system_prompt_rides_first_user_turn_when_history_starts_with_model():
    # History-first message list: assistant (model) turn comes first.
    messages = [
        {"role": "assistant", "content": "earlier reply"},
        {"role": "user", "content": "follow-up question"},
    ]
    contents = _assemble(messages, system_prompt="SYS GUIDANCE")

    # The leading turn is still the model history turn -- system prompt must
    # NOT have been attached to it.
    assert contents[0].role == "model"
    assert all(p.text != "SYS GUIDANCE" for p in contents[0].parts)

    # The first user-role turn carries the system prompt, prepended.
    user_turn = next(c for c in contents if c.role == "user")
    assert user_turn.parts[0].text == "SYS GUIDANCE"
    assert user_turn.parts[1].text == "follow-up question"


def test_system_prompt_prepended_to_first_user_turn_simple():
    messages = [{"role": "user", "content": "hello"}]
    contents = _assemble(messages, system_prompt="SYS")

    assert len(contents) == 1
    assert contents[0].role == "user"
    assert [p.text for p in contents[0].parts] == ["SYS", "hello"]


def test_no_user_turn_inserts_new_leading_user_content():
    # Only a model turn exists -> a NEW leading user turn must be inserted.
    messages = [{"role": "assistant", "content": "only a model turn"}]
    contents = _assemble(messages, system_prompt="SYS")

    assert contents[0].role == "user"
    assert [p.text for p in contents[0].parts] == ["SYS"]
    assert contents[1].role == "model"


def test_empty_messages_with_system_prompt_inserts_user_turn():
    contents = _assemble([], system_prompt="SYS")
    assert len(contents) == 1
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "SYS"


def test_no_system_prompt_leaves_contents_untouched():
    messages = [
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "b"},
    ]
    contents = _assemble(messages, system_prompt=None)

    assert [c.role for c in contents] == ["model", "user"]
    assert contents[0].parts[0].text == "a"
    assert contents[1].parts[0].text == "b"
