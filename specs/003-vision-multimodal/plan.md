# Vision (Image Understanding) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user attach images to a chat turn (web UI + Telegram/Discord/Slack/Teams); OpsRAG sends them to a vision-capable LLM alongside the question for that turn, ephemerally, auto-routing to a vision model when the active model can't see.

**Architecture:** Image bytes ride in the LangGraph runnable `config` (`config["configurable"]["turn_images"]`), never in graph `state` — so the Postgres checkpointer never persists them. A new neutral content format (`str` OR a parts list) is converted per-provider into native multimodal message blocks. The generator node reads the images at generation time and routes to a configured vision model if the active model isn't vision-capable.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, LangGraph, pytest + pytest-asyncio; React 19 + Vite (UI). Providers: Anthropic, Bedrock (Converse), Vertex (Claude + Gemini), OpenAI, LiteLLM.

**Spec:** `specs/003-vision-multimodal/spec.md`

**Test run convention (in-container, matches this repo):**
```bash
docker run --rm -v "$PWD":/app -w /app --entrypoint sh \
  ghcr.io/dtduc-git/opsrag:latest -c \
  'pip install -q pytest pytest-asyncio && \
   export PYTHONPATH=/home/opsrag/.local/lib/python3.11/site-packages && \
   python -m pytest <TEST_PATH> -v'
```
If a local venv with deps exists, `python -m pytest <TEST_PATH> -v` is fine. Each task's `Run:` lines show the `<TEST_PATH>`.

---

## File Structure

**New files:**
- `opsrag/llms/content.py` — `ImagePart`, neutral content builders, per-provider converters, vision-capability map, provider-aware default-vision-model resolver. *One responsibility: the multimodal content contract.*
- `tests/llms/test_content.py` — unit tests for the above.
- `tests/agent/test_vision_ephemeral.py` — the ephemerality guarantee (SC-002).

**Modified:**
- `opsrag/config.py` — add `VisionConfig`; attach to the top-level config.
- `opsrag/factory.py` — build `vision_llm`; expose on the providers container.
- `opsrag/llms/{anthropic,bedrock,openai,litellm_provider,vertex}.py` — run message content through the converter.
- `opsrag/agent/graph.py` — `query_with_session` + `query_with_session_events` gain `images` + `vision_llm`; injected into `config`.
- `opsrag/agent/nodes/generator.py` — consume `turn_images` from `config`; vision routing.
- `opsrag/api/models.py` — `ImageInput`; `QueryRequest.images`.
- `opsrag/api/routes.py` — decode/validate images; thread to the agent.
- `opsrag/channels/types.py` — `ImageRef`; `InboundMessage.images`.
- `opsrag/channels/interfaces.py` (the `ChannelAdapter` Protocol) — add `fetch_image`.
- `opsrag/channels/dispatcher.py` — fetch-after-permission; pass images.
- `opsrag/channels/adapters/{telegram,discord,slack,teams}/…` — extract `ImageRef`; implement `fetch_image`.
- `ui/src/components/ChatInput.tsx`, `ui/src/api.ts`, `ui/src/components/ChatMessage.tsx` — attach/paste/drop + send + render.

---

## Task 1: Neutral content model + per-provider converters

**Files:**
- Create: `opsrag/llms/content.py`
- Test: `tests/llms/test_content.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/llms/test_content.py
import base64
from opsrag.llms.content import (
    ImagePart, build_user_content, is_multimodal,
    to_anthropic_content, to_bedrock_content, to_openai_content,
)

PNG = b"\x89PNG\r\n\x1a\nFAKEBYTES"


def test_no_images_returns_plain_string_fast_path():
    assert build_user_content("hello", []) == "hello"
    assert is_multimodal("hello") is False


def test_with_images_returns_parts_list():
    content = build_user_content("look", [ImagePart(PNG, "image/png", "a.png")])
    assert is_multimodal(content) is True
    assert content[0] == {"type": "text", "text": "look"}
    assert content[1]["type"] == "image"
    assert content[1]["mime_type"] == "image/png"
    assert content[1]["data"] == PNG


def test_anthropic_converts_image_to_base64_block():
    content = build_user_content("q", [ImagePart(PNG, "image/png")])
    out = to_anthropic_content(content)
    assert out[0] == {"type": "text", "text": "q"}
    assert out[1]["type"] == "image"
    assert out[1]["source"]["type"] == "base64"
    assert out[1]["source"]["media_type"] == "image/png"
    assert out[1]["source"]["data"] == base64.b64encode(PNG).decode("ascii")


def test_anthropic_passthrough_string():
    assert to_anthropic_content("plain") == "plain"


def test_bedrock_wraps_string_in_text_block():
    assert to_bedrock_content("plain") == [{"text": "plain"}]


def test_bedrock_converts_image_to_format_and_raw_bytes():
    content = build_user_content("q", [ImagePart(PNG, "image/jpeg")])
    out = to_bedrock_content(content)
    assert out[0] == {"text": "q"}
    assert out[1]["image"]["format"] == "jpeg"
    assert out[1]["image"]["source"]["bytes"] == PNG  # raw bytes, NOT base64


def test_openai_converts_image_to_data_url():
    content = build_user_content("q", [ImagePart(PNG, "image/webp")])
    out = to_openai_content(content)
    assert out[0] == {"type": "text", "text": "q"}
    b64 = base64.b64encode(PNG).decode("ascii")
    assert out[1] == {
        "type": "image_url",
        "image_url": {"url": f"data:image/webp;base64,{b64}"},
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llms/test_content.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'opsrag.llms.content'`

- [ ] **Step 3: Write minimal implementation**

```python
# opsrag/llms/content.py
"""Multimodal message-content contract shared by every LLM provider.

OpsRAG message dicts are ``{"role": str, "content": <content>}`` where
``content`` is either a plain ``str`` (the text-only fast path, unchanged for
every existing call site) OR a provider-neutral *parts list*:

    [{"type": "text", "text": str},
     {"type": "image", "mime_type": str, "data": bytes}]

Each provider's ``generate()`` runs message content through the matching
``to_<provider>_content`` converter so the rest of the codebase only ever
deals with the neutral form. Image bytes are EPHEMERAL — they live only for
the turn and are never persisted (see spec FR-003).
"""
from __future__ import annotations

import base64
from dataclasses import dataclass


@dataclass(frozen=True)
class ImagePart:
    """A single in-memory image for one turn. Never serialized to a store."""

    data: bytes
    mime_type: str
    name: str | None = None


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def image_part(img: ImagePart) -> dict:
    return {"type": "image", "mime_type": img.mime_type, "data": img.data}


def build_user_content(text: str, images: list[ImagePart] | None):
    """Plain ``str`` when there are no images (fast path), else a parts list."""
    if not images:
        return text
    parts: list[dict] = []
    if text:
        parts.append(text_part(text))
    for img in images:
        parts.append(image_part(img))
    return parts


def is_multimodal(content) -> bool:
    return isinstance(content, list)


# ---- provider converters: neutral content (str | list) -> native shape ----

_BEDROCK_FORMATS = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/gif": "gif",
    "image/webp": "webp",
}


def to_anthropic_content(content):
    """Anthropic / Anthropic-on-Vertex content blocks."""
    if not is_multimodal(content):
        return content
    out: list[dict] = []
    for p in content:
        if p["type"] == "text":
            out.append({"type": "text", "text": p["text"]})
        elif p["type"] == "image":
            out.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": p["mime_type"],
                    "data": base64.b64encode(p["data"]).decode("ascii"),
                },
            })
    return out


def to_bedrock_content(content):
    """Bedrock Converse content blocks (always a list)."""
    if not is_multimodal(content):
        return [{"text": content}]
    out: list[dict] = []
    for p in content:
        if p["type"] == "text":
            out.append({"text": p["text"]})
        elif p["type"] == "image":
            fmt = _BEDROCK_FORMATS.get(p["mime_type"], "png")
            out.append({"image": {"format": fmt, "source": {"bytes": p["data"]}}})
    return out


def to_openai_content(content):
    """OpenAI / LiteLLM content array (data-URL images)."""
    if not is_multimodal(content):
        return content
    out: list[dict] = []
    for p in content:
        if p["type"] == "text":
            out.append({"type": "text", "text": p["text"]})
        elif p["type"] == "image":
            b64 = base64.b64encode(p["data"]).decode("ascii")
            out.append({
                "type": "image_url",
                "image_url": {"url": f"data:{p['mime_type']};base64,{b64}"},
            })
    return out


def to_gemini_parts(content):
    """A list of ``vertexai.generative_models.Part`` for one message's content."""
    from vertexai.generative_models import Part

    if not is_multimodal(content):
        return [Part.from_text(content)]
    parts = []
    for p in content:
        if p["type"] == "text":
            parts.append(Part.from_text(p["text"]))
        elif p["type"] == "image":
            parts.append(Part.from_data(mime_type=p["mime_type"], data=p["data"]))
    return parts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llms/test_content.py -v`
Expected: PASS (7 passed). `to_gemini_parts` isn't tested here (needs the Vertex SDK) — covered indirectly in Task 4e.

- [ ] **Step 5: Commit**

```bash
git add opsrag/llms/content.py tests/llms/test_content.py
git commit -m "feat(vision): neutral multimodal content model + provider converters"
```

---

## Task 2: Vision capability map + provider-aware default model

**Files:**
- Modify: `opsrag/llms/content.py`
- Test: `tests/llms/test_content.py`

- [ ] **Step 1: Write the failing test (append to the existing test file)**

```python
# tests/llms/test_content.py  (append)
from opsrag.llms.content import is_vision_capable, default_vision_model


def test_vision_capable_known_models():
    assert is_vision_capable("anthropic", "claude-sonnet-4-20250514") is True
    assert is_vision_capable("bedrock", "anthropic.claude-sonnet-4-20250514-v1:0") is True
    assert is_vision_capable("vertex", "gemini-3-flash-preview") is True
    assert is_vision_capable("openai", "gpt-4o") is True


def test_vision_incapable_models():
    assert is_vision_capable("openai", "gpt-3.5-turbo") is False
    assert is_vision_capable("litellm", "mistral/mistral-small") is False
    assert is_vision_capable("anthropic", "") is False


def test_default_vision_model_per_provider():
    assert default_vision_model("vertex") == "gemini-3-flash-preview"
    assert "claude" in default_vision_model("anthropic")
    assert "claude" in default_vision_model("bedrock")
    assert default_vision_model("litellm") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llms/test_content.py -k vision_capable -v`
Expected: FAIL — `ImportError: cannot import name 'is_vision_capable'`

- [ ] **Step 3: Add the implementation to `opsrag/llms/content.py`**

```python
# opsrag/llms/content.py  (append)

# Substrings that mark a vision-capable model id, matched case-insensitively
# against the model string regardless of provider prefixing. Conservative —
# add families as they ship. Operators can always force a vision model via
# `vision.model` / OPSRAG_VISION_MODEL when a new id isn't recognised here.
_VISION_MODEL_MARKERS = (
    "claude-3", "claude-4", "claude-opus-4", "claude-sonnet-4", "claude-haiku-4",
    "opus-4", "sonnet-4", "haiku-4",
    "gpt-4o", "gpt-4.1", "gpt-5", "o4-",
    "gemini-1.5", "gemini-2", "gemini-3",
)

# Provider-aware default when the active model can't see and no explicit
# `vision.model` is configured (spec FR-011). These ids are valid in this
# codebase out of the box; for Sonnet 4.6 specifically set OPSRAG_VISION_MODEL
# / vision.model to your exact id.
_DEFAULT_VISION_MODEL = {
    "anthropic": "claude-sonnet-4-20250514",
    "bedrock": "anthropic.claude-sonnet-4-20250514-v1:0",
    "vertex": "gemini-3-flash-preview",
    "openai": "gpt-4o",
    "litellm": None,
}


def is_vision_capable(provider: str, model: str | None) -> bool:
    m = (model or "").lower()
    if not m:
        return False
    return any(marker in m for marker in _VISION_MODEL_MARKERS)


def default_vision_model(provider: str) -> str | None:
    return _DEFAULT_VISION_MODEL.get(provider)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llms/test_content.py -v`
Expected: PASS (all, incl. the 3 new tests).

- [ ] **Step 5: Commit**

```bash
git add opsrag/llms/content.py tests/llms/test_content.py
git commit -m "feat(vision): vision-capability map + provider-aware default model"
```

---

## Task 3: `VisionConfig`

**Files:**
- Modify: `opsrag/config.py` (add class near `LLMConfig` at line ~132–147; attach to the top-level config class)
- Test: `tests/test_config_vision.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_vision.py
from opsrag.config import OpsRAGConfig, VisionConfig


def test_vision_defaults():
    v = VisionConfig()
    assert v.enabled is True
    assert v.max_images == 4
    assert v.max_bytes == 5 * 1024 * 1024
    assert set(v.allowed_mime) == {"image/png", "image/jpeg", "image/gif", "image/webp"}
    assert v.model is None
    assert v.provider is None


def test_top_level_config_has_vision_section():
    cfg = OpsRAGConfig()
    assert isinstance(cfg.vision, VisionConfig)
```

> If the top-level class is not named `OpsRAGConfig`, open `opsrag/config.py`, find the root `BaseModel` that aggregates `llm: LLMConfig`, and use that name in both the test and Step 3.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_vision.py -v`
Expected: FAIL — `ImportError: cannot import name 'VisionConfig'`

- [ ] **Step 3: Add `VisionConfig` after `LLMConfig` (after line 147) and attach it**

```python
# opsrag/config.py  (insert after the LLMConfig class, ~line 148)
class VisionConfig(BaseModel):
    """Image/vision behaviour. All overridable via env/YAML (no rebuild).

    `model`/`provider` are the auto-route fallback used ONLY when an image
    arrives and the active model isn't vision-capable; left None, a
    provider-aware default is resolved at factory time
    (opsrag.llms.content.default_vision_model). Bytes are ephemeral — never
    persisted (spec FR-003).
    """

    enabled: bool = True
    model: str | None = None
    provider: Literal["anthropic", "openai", "vertex", "bedrock", "litellm"] | None = None
    max_images: int = 4
    max_bytes: int = 5 * 1024 * 1024
    allowed_mime: list[str] = Field(
        default_factory=lambda: ["image/png", "image/jpeg", "image/gif", "image/webp"]
    )
```

Then add a field to the top-level config class (the one that already has `llm: LLMConfig = Field(default_factory=LLMConfig)`):

```python
    vision: VisionConfig = Field(default_factory=VisionConfig)
```

> `Field` and `Literal` are already imported in `config.py` (used by `LLMConfig`). If `Field` is missing, add `from pydantic import BaseModel, Field`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config_vision.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add opsrag/config.py tests/test_config_vision.py
git commit -m "feat(vision): VisionConfig (limits, mime allow-list, auto-route model)"
```

---

## Task 4: Wire converters into the 5 providers

Each sub-task changes one provider's `generate()` to convert per-message content. All keep the text-only fast path byte-identical.

### Task 4a — Anthropic

**Files:**
- Modify: `opsrag/llms/anthropic.py:51-56` (the `kwargs["messages"]` build)
- Test: `tests/llms/test_provider_multimodal.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/llms/test_provider_multimodal.py
from opsrag.llms.anthropic import AnthropicLLM
from opsrag.llms.content import build_user_content, ImagePart

PNG = b"\x89PNG\r\nfake"


def _msgs():
    return [{"role": "user", "content": build_user_content("q", [ImagePart(PNG, "image/png")])}]


def test_anthropic_builds_image_block(monkeypatch):
    captured = {}

    class _Resp:
        class _Usage:
            input_tokens = 1
            output_tokens = 1
        content = []
        model = "claude-sonnet-4-20250514"
        usage = _Usage()

    class _Messages:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return _Resp()

    class _Client:
        messages = _Messages()

    llm = AnthropicLLM(api_key="x", model="claude-sonnet-4-20250514")
    monkeypatch.setattr(llm, "_get_client", lambda: _Client())

    import asyncio
    asyncio.run(llm.generate(messages=_msgs()))

    sent = captured["messages"][0]["content"]
    assert isinstance(sent, list)
    assert sent[1]["type"] == "image"
    assert sent[1]["source"]["media_type"] == "image/png"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llms/test_provider_multimodal.py::test_anthropic_builds_image_block -v`
Expected: FAIL — `sent` is the raw neutral list (`{"type":"image","mime_type":...}`), so `sent[1]["source"]` raises `KeyError`.

- [ ] **Step 3: Convert content before the SDK call**

In `opsrag/llms/anthropic.py`, add the import near the top (after line 11):

```python
from opsrag.llms.content import to_anthropic_content
```

Replace the `kwargs` build (lines 51-56) so messages are converted:

```python
        converted = [
            {"role": m["role"], "content": to_anthropic_content(m.get("content", ""))}
            for m in messages
        ]
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": converted,
            "max_tokens": max_tokens or self._default_max_tokens,
            "temperature": temperature,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llms/test_provider_multimodal.py::test_anthropic_builds_image_block -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add opsrag/llms/anthropic.py tests/llms/test_provider_multimodal.py
git commit -m "feat(vision): anthropic provider converts multimodal content"
```

### Task 4b — Bedrock

**Files:**
- Modify: `opsrag/llms/bedrock.py:72-77`
- Test: `tests/llms/test_provider_multimodal.py`

- [ ] **Step 1: Add the failing test**

```python
# tests/llms/test_provider_multimodal.py  (append)
def test_bedrock_builds_image_block(monkeypatch):
    from opsrag.llms.bedrock import BedrockLLM
    captured = {}

    def _converse(**kwargs):
        captured.update(kwargs)
        return {"output": {"message": {"content": [{"text": "ok"}]}},
                "usage": {"inputTokens": 1, "outputTokens": 1}}

    llm = BedrockLLM.__new__(BedrockLLM)            # skip boto3 __init__
    llm._client = type("C", (), {"converse": staticmethod(_converse)})()
    llm._model = "anthropic.claude-sonnet-4-20250514-v1:0"
    llm._default_max_tokens = 4096

    import asyncio
    asyncio.run(llm.generate(messages=_msgs()))

    sent = captured["messages"][0]["content"]
    assert sent[1]["image"]["format"] == "png"
    assert sent[1]["image"]["source"]["bytes"] == PNG
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llms/test_provider_multimodal.py::test_bedrock_builds_image_block -v`
Expected: FAIL — current code does `{"text": msg["content"]}` where `content` is a list → the dict has no `image` block.

- [ ] **Step 3: Use the converter**

In `opsrag/llms/bedrock.py`, add import (after line 23):

```python
from opsrag.llms.content import to_bedrock_content
```

Replace the loop at lines 72-77:

```python
        converse_messages = []
        for msg in messages:
            converse_messages.append({
                "role": msg["role"],
                "content": to_bedrock_content(msg.get("content", "")),
            })
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llms/test_provider_multimodal.py::test_bedrock_builds_image_block -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add opsrag/llms/bedrock.py tests/llms/test_provider_multimodal.py
git commit -m "feat(vision): bedrock provider converts multimodal content"
```

### Task 4c — OpenAI

**Files:**
- Modify: `opsrag/llms/openai.py:65-68`
- Test: `tests/llms/test_provider_multimodal.py`

- [ ] **Step 1: Add the failing test**

```python
# tests/llms/test_provider_multimodal.py  (append)
def test_openai_builds_image_url(monkeypatch):
    from opsrag.llms.openai import OpenAILLM
    captured = {}

    class _Msg:
        content = "ok"
    class _Choice:
        message = _Msg()
    class _Usage:
        prompt_tokens = 1
        completion_tokens = 1
    class _Resp:
        choices = [_Choice()]
        usage = _Usage()
        model = "gpt-4o"
    class _Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return _Resp()
    class _Chat:
        completions = _Completions()
    class _Client:
        chat = _Chat()

    llm = OpenAILLM(api_key="x", model="gpt-4o")
    monkeypatch.setattr(llm, "_get_client", lambda: _Client())

    import asyncio
    asyncio.run(llm.generate(messages=_msgs()))

    sent = captured["messages"][-1]["content"]      # last = the user message
    assert sent[1]["type"] == "image_url"
    assert sent[1]["image_url"]["url"].startswith("data:image/png;base64,")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llms/test_provider_multimodal.py::test_openai_builds_image_url -v`
Expected: FAIL — content is the raw neutral list; `sent[1]["type"]` is `"image"` not `"image_url"`.

- [ ] **Step 3: Convert each message's content**

In `opsrag/llms/openai.py`, add import (after line 17):

```python
from opsrag.llms.content import to_openai_content
```

Replace lines 65-68:

```python
        full_messages: list[dict] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(
            {"role": m["role"], "content": to_openai_content(m.get("content", ""))}
            for m in messages
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llms/test_provider_multimodal.py::test_openai_builds_image_url -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add opsrag/llms/openai.py tests/llms/test_provider_multimodal.py
git commit -m "feat(vision): openai provider converts multimodal content"
```

### Task 4d — LiteLLM

**Files:**
- Modify: `opsrag/llms/litellm_provider.py:70-73`
- Test: `tests/llms/test_provider_multimodal.py`

- [ ] **Step 1: Add the failing test**

```python
# tests/llms/test_provider_multimodal.py  (append)
def test_litellm_builds_image_url(monkeypatch):
    import sys, types
    captured = {}

    async def _acompletion(**kwargs):
        captured.update(kwargs)
        class _Msg: content = "ok"
        class _Choice: message = _Msg()
        class _Usage: prompt_tokens = 1; completion_tokens = 1
        class _Resp:
            choices = [_Choice()]; usage = _Usage(); model = "gemini/gemini-3-flash-preview"
        return _Resp()

    fake = types.ModuleType("litellm")
    fake.acompletion = _acompletion
    monkeypatch.setitem(sys.modules, "litellm", fake)

    from opsrag.llms.litellm_provider import LiteLLMLLM
    llm = LiteLLMLLM(model="gemini/gemini-3-flash-preview")

    import asyncio
    asyncio.run(llm.generate(messages=_msgs()))

    sent = captured["messages"][-1]["content"]
    assert sent[1]["type"] == "image_url"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llms/test_provider_multimodal.py::test_litellm_builds_image_url -v`
Expected: FAIL — raw neutral content passes straight through; `sent[1]["type"]` is `"image"`.

- [ ] **Step 3: Convert in the `generate()` message build**

In `opsrag/llms/litellm_provider.py`, add import (after line 28):

```python
from opsrag.llms.content import to_openai_content
```

Replace lines 70-73 (inside `generate`, NOT `_to_openai_messages`):

```python
        full_messages: list[dict] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(
            {"role": m["role"], "content": to_openai_content(m.get("content", ""))}
            for m in messages
        )
```

> Leave `_to_openai_messages` (the tool-loop translator, lines 125-179) untouched — tool turns are always text.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llms/test_provider_multimodal.py::test_litellm_builds_image_url -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add opsrag/llms/litellm_provider.py tests/llms/test_provider_multimodal.py
git commit -m "feat(vision): litellm provider converts multimodal content"
```

### Task 4e — Vertex (Claude + Gemini)

**Files:**
- Modify: `opsrag/llms/vertex.py:315-322` (Claude path) and `:350-376` (Gemini path)
- Test: `tests/llms/test_provider_multimodal.py`

- [ ] **Step 1: Add the failing test (Claude-on-Vertex path; Gemini needs the SDK so we assert via the converter)**

```python
# tests/llms/test_provider_multimodal.py  (append)
def test_vertex_claude_builds_image_block(monkeypatch):
    from opsrag.llms.vertex import VertexAILLM

    captured = {}

    class _Resp:
        class _Usage: input_tokens = 1; output_tokens = 1
        content = []; model = "claude-sonnet-4@20250514"; usage = _Usage()

    def _create(**kwargs):
        captured.update(kwargs)
        return _Resp()

    llm = VertexAILLM.__new__(VertexAILLM)
    llm._model = "claude-sonnet-4@20250514"
    llm._is_claude = True
    llm._project = "p"; llm._location = "us-east5"
    llm._client = type("C", (), {"messages": type("M", (), {"create": staticmethod(_create)})()})()
    llm._fire_on_usage = lambda **k: _noop()

    import asyncio
    async def _noop(): return None
    monkeypatch.setattr(llm, "_get_client", lambda: llm._client)
    monkeypatch.setattr(llm, "_fire_on_usage", lambda **k: _noop())

    asyncio.run(llm.generate(messages=_msgs()))
    sent = captured["messages"][0]["content"]
    assert sent[1]["type"] == "image" and sent[1]["source"]["type"] == "base64"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/llms/test_provider_multimodal.py::test_vertex_claude_builds_image_block -v`
Expected: FAIL — Claude-on-Vertex passes `messages` through unconverted.

- [ ] **Step 3: Convert in both Vertex paths**

In `opsrag/llms/vertex.py`, add import near the existing imports at the top of the file:

```python
from opsrag.llms.content import to_anthropic_content, to_gemini_parts
```

In `_generate_claude` (lines 315-322) convert messages before building `kwargs`:

```python
        converted = [
            {"role": m["role"], "content": to_anthropic_content(m.get("content", ""))}
            for m in messages
        ]
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": converted,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
```

In `_generate_gemini`, replace the string-concatenation prompt build (lines 350-354 and the `generate_content` call at 372-376). Gemini takes a list of `Content`; build one user `Content` carrying text + image Parts, with the system prompt prepended as a leading text Part:

```python
        from vertexai.generative_models import Content, Part

        client = self._get_client()
        sys_parts = [Part.from_text(system_prompt)] if system_prompt else []
        contents = []
        for msg in messages:
            role = "model" if msg.get("role") == "assistant" else "user"
            parts = to_gemini_parts(msg.get("content", ""))
            contents.append(Content(role=role, parts=parts))
        # Prepend the system prompt to the first user turn's parts (Gemini has
        # no separate system role on this SDK path).
        if sys_parts and contents:
            contents[0] = Content(role=contents[0].role, parts=sys_parts + contents[0].parts)
```

Then change the `generate_content` call (line 372-376) to pass `contents` instead of the joined string:

```python
        resp = await asyncio.to_thread(
            client.generate_content,
            contents,
            generation_config=config,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/llms/test_provider_multimodal.py::test_vertex_claude_builds_image_block -v`
Expected: PASS. (Gemini path is exercised end-to-end in manual verification; its conversion is unit-tested via `to_gemini_parts` shape in Task 1's module — if `vertexai` is unavailable in CI the import is lazy so it won't break collection.)

- [ ] **Step 5: Commit**

```bash
git add opsrag/llms/vertex.py tests/llms/test_provider_multimodal.py
git commit -m "feat(vision): vertex claude+gemini providers convert multimodal content"
```

---

## Task 5: Build `vision_llm` in the factory

**Files:**
- Modify: `opsrag/factory.py` (after the `purpose_router` build at line 285; and the providers container that exposes `.llm`)
- Test: `tests/test_factory_vision.py`

> First locate the providers container: in `factory.py`, find where the object that the dispatcher reads as `self._providers.llm` is constructed (it has an `llm` attribute). Add a `vision_llm` attribute to it. The steps below assume that object is assembled near the end of `build_providers` (or equivalent); adapt the attribute assignment to the actual container.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_factory_vision.py
from opsrag.llms.content import is_vision_capable, default_vision_model
from opsrag.config import VisionConfig, LLMConfig


def _resolve(vision: VisionConfig, llm: LLMConfig):
    # Mirror the factory's resolution rule (pure function under test).
    from opsrag.factory import resolve_vision_model
    return resolve_vision_model(vision, llm)


def test_resolve_uses_explicit_override():
    v = VisionConfig(model="claude-sonnet-4-6", provider="bedrock")
    out = _resolve(v, LLMConfig(provider="bedrock", model="some-text-only"))
    assert out == ("bedrock", "claude-sonnet-4-6")


def test_resolve_reuses_active_model_when_vision_capable():
    v = VisionConfig()
    llm = LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514")
    # active model already sees → no separate vision model needed
    assert _resolve(v, llm) is None


def test_resolve_falls_back_to_provider_default():
    v = VisionConfig()
    llm = LLMConfig(provider="vertex", model="text-bison")   # not vision-capable
    assert _resolve(v, llm) == ("vertex", "gemini-3-flash-preview")


def test_resolve_disabled_returns_none():
    assert _resolve(VisionConfig(enabled=False), LLMConfig()) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_factory_vision.py -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_vision_model'`

- [ ] **Step 3: Add `resolve_vision_model` + build the vision LLM**

Add this module-level helper to `opsrag/factory.py` (near the other helpers):

```python
def resolve_vision_model(vision, llm_cfg) -> tuple[str, str] | None:
    """Return (provider, model) for the vision fallback LLM, or None when no
    separate vision client is needed (disabled, or the active model already
    sees). Explicit vision.model/provider always wins (spec FR-011)."""
    from opsrag.llms.content import default_vision_model, is_vision_capable

    if not getattr(vision, "enabled", True):
        return None
    if vision.model:
        return (vision.provider or llm_cfg.provider, vision.model)
    if is_vision_capable(llm_cfg.provider, llm_cfg.model):
        return None  # active model can already see; generator reuses it
    provider = vision.provider or llm_cfg.provider
    model = default_vision_model(provider)
    return (provider, model) if model else None
```

After the `purpose_router` build (line 285), build the client and a tiny builder that reuses the same provider construction. Add a reusable provider builder (extract or inline). Minimal inline version:

```python
    # Vision fallback LLM (only when the active model can't see). Built once at
    # startup so per-turn vision routing costs no client setup.
    vision_llm: LLMProvider | None = None
    _vision_target = resolve_vision_model(config.vision, config.llm)
    if _vision_target is not None:
        v_provider, v_model = _vision_target
        if v_provider == "anthropic":
            vision_llm = AnthropicLLM(
                api_key=_env(config.llm.api_key_env),
                model=v_model,
                default_max_tokens=config.llm.max_tokens,
            )
        elif v_provider == "bedrock":
            from opsrag.llms.bedrock import BedrockLLM
            vision_llm = BedrockLLM(
                model=v_model,
                region=config.llm.aws_region,
                profile=config.llm.aws_profile,
                default_max_tokens=config.llm.max_tokens,
            )
        elif v_provider == "vertex":
            from opsrag.llms.vertex import VertexAILLM
            vision_llm = VertexAILLM(
                model=v_model,
                project=config.llm.project,
                location=config.llm.location or "us-central1",
                default_max_tokens=config.llm.max_tokens,
            )
        elif v_provider == "openai":
            from opsrag.llms.openai import OpenAILLM
            vision_llm = OpenAILLM(
                api_key=_env(config.llm.api_key_env),
                model=v_model,
                default_max_tokens=config.llm.max_tokens,
            )
        elif v_provider == "litellm":
            from opsrag.llms.litellm_provider import LiteLLMLLM
            vision_llm = LiteLLMLLM(
                model=v_model,
                default_max_tokens=config.llm.max_tokens,
                api_base=config.llm.api_base,
                api_key_env=config.llm.api_key_env,
            )
```

Then attach `vision_llm` to the providers container (the object exposing `.llm`). Find that assignment and add, e.g.:

```python
    providers.vision_llm = vision_llm   # adapt to the actual container/ctor
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_factory_vision.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add opsrag/factory.py tests/test_factory_vision.py
git commit -m "feat(vision): factory builds provider-aware vision fallback LLM"
```

---

## Task 6: Thread `images` + `vision_llm` through the agent entry points (into `config`, not `state`)

**Files:**
- Modify: `opsrag/agent/graph.py` — `query_with_session` (signature ~736-750, initial ~889, invoke ~980) and `query_with_session_events` (signature ~1117-1131, initial ~1264, config build before `astream_events` ~1328)
- Test: `tests/agent/test_vision_ephemeral.py`

- [ ] **Step 1: Write the failing test (asserts images go to config, NOT state)**

```python
# tests/agent/test_vision_ephemeral.py
import asyncio
from opsrag.llms.content import ImagePart

PNG = b"\x89PNG\r\nfake"


class _FakeGraph:
    """Captures the config + initial state passed to astream_events."""
    def __init__(self):
        self.seen_config = None
        self.seen_initial = None

    async def astream_events(self, initial, config=None, version=None):
        self.seen_initial = initial
        self.seen_config = config
        yield {"event": "on_chain_end", "name": "__end__",
               "data": {"output": {"generation": "ok", "final_chunks": []}}}


def test_images_ride_in_config_not_state():
    from opsrag.agent.graph import query_with_session_events
    g = _FakeGraph()

    async def _run():
        async for _ in query_with_session_events(
            g, query="look", user_id="u1", thread_id="web:u1:t1",
            images=[ImagePart(PNG, "image/png", "a.png")],
            vision_llm=object(),
        ):
            pass

    asyncio.run(_run())

    # Ephemeral guarantee: bytes are NOT in the checkpointed state...
    assert "turn_images" not in (g.seen_initial or {})
    assert PNG not in repr(g.seen_initial)
    # ...they live in the runnable config instead.
    cfg = g.seen_config["configurable"]
    assert cfg["turn_images"][0].data == PNG
    assert cfg["vision_llm"] is not None
    # The persisted query carries only a text marker, never bytes.
    assert "[attached image" in g.seen_initial["query"]
```

> The real `query_with_session_events` does much more than this fake exercises (cache, session store, event translation). The test injects a minimal fake graph and only asserts the config/state split. If the function accesses other providers before streaming, pass them as `None` (already defaulted) — the function must tolerate that for this path. If early code paths block on `None` providers, guard the new logic to run regardless and place the `images`/`vision_llm` plumbing immediately around the `config`/`initial` construction.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agent/test_vision_ephemeral.py::test_images_ride_in_config_not_state -v`
Expected: FAIL — `query_with_session_events` has no `images` / `vision_llm` parameters → `TypeError`.

- [ ] **Step 3: Add the parameters + the config injection**

In BOTH `query_with_session` and `query_with_session_events`, add to the signature (after `user_name`):

```python
    images: list[ImagePart] | None = None,
    vision_llm=None,
```

Add the import at the top of `graph.py`:

```python
from opsrag.llms.content import ImagePart
```

Where the query string is set into `initial` (line ~889 / ~1264), append a text marker when images are present so history/checkpoint records the fact, not the bytes:

```python
    query_for_state = query
    if images:
        names = ", ".join(img.name or "image" for img in images)
        query_for_state = f"{query} [attached image: {names}]".strip()
    initial: dict = {
        "query": query_for_state,
        ...
    }
```

Where the run `config` is built (the `config=` passed to `ainvoke` / `astream_events`), add the ephemeral side-channel under `configurable`:

```python
    config = {
        "configurable": {
            "thread_id": thread_id,
            # ... existing keys ...
            "turn_images": images or [],
            "vision_llm": vision_llm,
        },
        # ... existing recursion_limit etc ...
    }
```

> If `config` is currently built inline inside the `ainvoke`/`astream_events` call, hoist it to a local `config = {...}` first, then pass `config=config`. Keep every existing `configurable` key.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/agent/test_vision_ephemeral.py::test_images_ride_in_config_not_state -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add opsrag/agent/graph.py tests/agent/test_vision_ephemeral.py
git commit -m "feat(vision): pass images via runnable config (ephemeral) + history marker"
```

---

## Task 7: Generator node consumes images + vision routing

**Files:**
- Modify: `opsrag/agent/nodes/generator.py:115` (signature `_generate(state)` → `_generate(state, config)`), `:196-201` (message build), `:224-229` (LLM call)
- Test: `tests/agent/test_generator_vision.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agent/test_generator_vision.py
import asyncio
from opsrag.agent.nodes.generator import generate_node
from opsrag.llms.content import ImagePart, is_multimodal

PNG = b"\x89PNG\r\nfake"


class _LLM:
    def __init__(self, name, vision):
        self._name = name; self._vision = vision; self.last_messages = None
    @property
    def model_name(self): return self._name
    async def generate(self, messages, system_prompt=None, temperature=0.0, purpose=None):
        self.last_messages = messages
        class _R: content = "ok"; model = self._name; usage = {}; latency_ms = 1.0
        return _R()


class _Obs:
    async def log_llm_call(self, **k): return None


def _state():
    return {"query": "what is this", "graded_chunks": [], "merged_results": [],
            "retrieved_chunks": [], "conversation_history": []}


def test_image_makes_user_content_multimodal_on_vision_model():
    text_llm = _LLM("claude-sonnet-4-20250514", vision=True)   # already sees
    node = generate_node(text_llm, _Obs())
    cfg = {"configurable": {"turn_images": [ImagePart(PNG, "image/png")], "vision_llm": None}}
    asyncio.run(node(_state(), cfg))
    user_content = text_llm.last_messages[-1]["content"]
    assert is_multimodal(user_content)
    assert any(p["type"] == "image" for p in user_content)


def test_routes_to_vision_llm_when_active_model_blind():
    blind = _LLM("text-bison", vision=False)
    seer = _LLM("gemini-3-flash-preview", vision=True)
    node = generate_node(blind, _Obs())
    cfg = {"configurable": {"turn_images": [ImagePart(PNG, "image/png")], "vision_llm": seer}}
    asyncio.run(node(_state(), cfg))
    assert seer.last_messages is not None          # vision LLM was used
    assert blind.last_messages is None


def test_no_vision_available_drops_image_and_notes():
    blind = _LLM("text-bison", vision=False)
    node = generate_node(blind, _Obs())
    cfg = {"configurable": {"turn_images": [ImagePart(PNG, "image/png")], "vision_llm": None}}
    out = asyncio.run(node(_state(), cfg))
    assert not is_multimodal(blind.last_messages[-1]["content"])   # text-only
    assert "can't read images" in out["generation"].lower() or \
           "cannot read images" in out["generation"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agent/test_generator_vision.py -v`
Expected: FAIL — `_generate` takes only `state` → `TypeError: _generate() takes 1 positional argument but 2 were given`.

- [ ] **Step 3: Implement consumption + routing**

In `opsrag/agent/nodes/generator.py`, add imports near the top:

```python
from opsrag.llms.content import build_user_content, is_vision_capable
```

Change the signature (line 115) and add image/routing logic. Replace the message build (lines 196-201) and the LLM call (lines 224-229):

```python
    async def _generate(state: dict, config: dict | None = None) -> dict:
        query = state["query"]
```

(everything from line 117 down to the `messages =` build stays the same, then:)

```python
        # ---- Vision: ephemeral images arrive via runnable config ----------
        configurable = (config or {}).get("configurable", {})
        turn_images = configurable.get("turn_images") or []
        vision_llm = configurable.get("vision_llm")
        active_llm = gen_llm
        vision_note = ""
        user_text = f"Context:\n{context_block}{graph_ctx}{tree_block}{anchor_hint}\n\nQuestion: {query}"
        user_content = user_text

        if turn_images:
            if is_vision_capable(
                getattr(active_llm, "provider_name", ""), active_llm.model_name
            ) or is_vision_capable("", active_llm.model_name):
                user_content = build_user_content(user_text, turn_images)
            elif vision_llm is not None:
                active_llm = vision_llm
                user_content = build_user_content(user_text, turn_images)
            else:
                vision_note = (
                    "\n\n_⚠️ Note: I can't read images with the current model, "
                    "so I answered from the text only._"
                )

        messages = history_msgs + [{"role": "user", "content": user_content}]
```

Then change the generate call to use `active_llm`:

```python
        response = await active_llm.generate(
            purpose="generation",
            messages=messages,
            system_prompt=system_prompt,
            temperature=gen_temp,
        )
```

And append the note to the returned generation:

```python
        return {
            "generation": response.content + vision_note,
            "current_step": "generated",
            "final_chunks": chunks,
        }
```

> The double `is_vision_capable(..., model_name)` call tolerates providers that don't expose `provider_name`; matching on the model id alone is sufficient because the markers are provider-agnostic.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/agent/test_generator_vision.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add opsrag/agent/nodes/generator.py tests/agent/test_generator_vision.py
git commit -m "feat(vision): generator consumes turn images + auto-routes to vision LLM"
```

---

## Task 8: Web API — `QueryRequest.images` + route validation

**Files:**
- Modify: `opsrag/api/models.py:7-11`
- Modify: `opsrag/api/routes.py` (the `query` handler ~589-660; pass `images`/`vision_llm` to `query_with_session*`)
- Test: `tests/api/test_query_images.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_query_images.py
import base64
import pytest
from pydantic import ValidationError
from opsrag.api.models import QueryRequest, ImageInput
from opsrag.api.images import decode_images, ImageValidationError
from opsrag.config import VisionConfig

PNG_B64 = base64.b64encode(b"\x89PNG\r\nfake").decode()


def test_query_request_accepts_images():
    req = QueryRequest(query="hi", images=[ImageInput(mime_type="image/png", data=PNG_B64)])
    assert req.images[0].mime_type == "image/png"


def test_query_request_images_optional():
    assert QueryRequest(query="hi").images is None


def test_decode_rejects_bad_mime():
    v = VisionConfig()
    with pytest.raises(ImageValidationError):
        decode_images([ImageInput(mime_type="application/pdf", data=PNG_B64)], v)


def test_decode_rejects_too_many():
    v = VisionConfig(max_images=1)
    imgs = [ImageInput(mime_type="image/png", data=PNG_B64),
            ImageInput(mime_type="image/png", data=PNG_B64)]
    with pytest.raises(ImageValidationError):
        decode_images(imgs, v)


def test_decode_rejects_oversize():
    v = VisionConfig(max_bytes=4)
    with pytest.raises(ImageValidationError):
        decode_images([ImageInput(mime_type="image/png", data=PNG_B64)], v)


def test_decode_ok_returns_image_parts():
    v = VisionConfig()
    parts = decode_images([ImageInput(mime_type="image/png", data=PNG_B64)], v)
    assert parts[0].mime_type == "image/png"
    assert parts[0].data == b"\x89PNG\r\nfake"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_query_images.py -v`
Expected: FAIL — `ImportError` for `ImageInput` / `opsrag.api.images`.

- [ ] **Step 3a: Add `ImageInput` + `QueryRequest.images`**

In `opsrag/api/models.py`, after the imports add nothing new (Field/BaseModel present). Insert before `QueryRequest`:

```python
class ImageInput(BaseModel):
    """A base64-encoded image attached to a chat turn (ephemeral)."""
    mime_type: str = Field(..., description="image/png | image/jpeg | image/gif | image/webp")
    data: str = Field(..., description="base64-encoded image bytes (no data: prefix)")
```

Add the field to `QueryRequest` (after `stream`):

```python
    images: list[ImageInput] | None = Field(
        None, description="Optional images for a vision-capable model (ephemeral)"
    )
```

- [ ] **Step 3b: Add the decoder/validator**

Create `opsrag/api/images.py`:

```python
"""Decode + validate web-attached images into ephemeral ImageParts."""
from __future__ import annotations

import base64
import binascii

from opsrag.api.models import ImageInput
from opsrag.config import VisionConfig
from opsrag.llms.content import ImagePart


class ImageValidationError(ValueError):
    """Raised when an attached image violates VisionConfig limits."""


def decode_images(images: list[ImageInput] | None, vision: VisionConfig) -> list[ImagePart]:
    if not images:
        return []
    if not vision.enabled:
        raise ImageValidationError("Image input is disabled on this deployment.")
    if len(images) > vision.max_images:
        raise ImageValidationError(
            f"Too many images: {len(images)} > max {vision.max_images}."
        )
    out: list[ImagePart] = []
    for i, img in enumerate(images):
        if img.mime_type not in vision.allowed_mime:
            raise ImageValidationError(f"Unsupported image type: {img.mime_type}.")
        try:
            raw = base64.b64decode(img.data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ImageValidationError(f"Image {i} is not valid base64.") from exc
        if len(raw) > vision.max_bytes:
            raise ImageValidationError(
                f"Image {i} too large: {len(raw)} bytes > max {vision.max_bytes}."
            )
        out.append(ImagePart(data=raw, mime_type=img.mime_type, name=f"image-{i}"))
    return out
```

- [ ] **Step 3c: Wire into the `query` route**

In `opsrag/api/routes.py`, inside the `query` handler, before calling the agent, decode + handle errors and pass through:

```python
    from opsrag.api.images import decode_images, ImageValidationError
    try:
        turn_images = decode_images(req.images, config.vision)
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
```

Then add `images=turn_images, vision_llm=<providers>.vision_llm` to BOTH the streaming `query_with_session_events(...)` and the non-streaming `query_with_session(...)` calls in this handler. Use the same providers object the handler already uses for `llm=` (search the handler for `llm=` to find the attribute path, e.g. `providers.vision_llm`).

> `config` (the loaded `OpsRAGConfig`) and the providers container are already in scope in this handler — reuse the same references the handler uses for the existing `llm=`/`session_store=` arguments.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/api/test_query_images.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add opsrag/api/models.py opsrag/api/images.py opsrag/api/routes.py tests/api/test_query_images.py
git commit -m "feat(vision): web QueryRequest.images + decode/validate + agent wiring"
```

---

## Task 9: Channel types — `ImageRef` + `InboundMessage.images` + adapter `fetch_image`

**Files:**
- Modify: `opsrag/channels/types.py:33-51`
- Modify: `opsrag/channels/interfaces.py` (the `ChannelAdapter` Protocol — add `fetch_image`)
- Test: `tests/channels/test_types_images.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/channels/test_types_images.py
from opsrag.channels.types import InboundMessage, ImageRef


def test_inbound_defaults_to_no_images():
    msg = InboundMessage(
        channel_id="c", user_id="u", text="hi", message_id="m",
        thread_id=None, is_dm=True, workspace=None,
    )
    assert msg.images == ()


def test_image_ref_fields():
    ref = ImageRef(file_id="F1", url="https://x/y.png", mime_type="image/png", size=10)
    assert ref.file_id == "F1"
    assert ref.mime_type == "image/png"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/channels/test_types_images.py -v`
Expected: FAIL — `ImportError: cannot import name 'ImageRef'`

- [ ] **Step 3: Add `ImageRef` + the field**

In `opsrag/channels/types.py`, after the imports add a frozen dataclass and extend `InboundMessage`:

```python
@dataclass(frozen=True)
class ImageRef:
    """A lightweight, pre-fetch reference to a platform image attachment.

    Adapters emit these without downloading; the dispatcher resolves them to
    bytes (via ``adapter.fetch_image``) only AFTER the permission check passes
    (spec FR-007). At least one of ``file_id`` / ``url`` is set.
    """

    file_id: str | None = None
    url: str | None = None
    mime_type: str = "image/png"
    size: int | None = None
```

Add to `InboundMessage` (after `workspace`, before `raw`):

```python
    images: tuple[ImageRef, ...] = field(default_factory=tuple)
```

> `field` is already imported. Place `images` BEFORE the existing `raw` field, or give it a default (it has one) — frozen dataclass fields with defaults must follow non-default fields; `raw` already has a default so order among the two defaulted fields is free.

- [ ] **Step 4a: Add `fetch_image` to the adapter Protocol**

In `opsrag/channels/interfaces.py`, find the `ChannelAdapter` Protocol and add:

```python
    async def fetch_image(self, ref: "ImageRef") -> bytes | None:
        """Download the bytes for an inbound image reference, or None on
        failure. Called only after the permission check passes."""
        ...
```

Add `ImageRef` to the import from `opsrag.channels.types` in that file.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/channels/test_types_images.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add opsrag/channels/types.py opsrag/channels/interfaces.py tests/channels/test_types_images.py
git commit -m "feat(vision): ImageRef + InboundMessage.images + adapter.fetch_image port"
```

---

## Task 10: Dispatcher — fetch-after-permission + pass images

**Files:**
- Modify: `opsrag/channels/dispatcher.py:171-256`
- Test: `tests/channels/test_dispatcher_images.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/channels/test_dispatcher_images.py
import asyncio
from opsrag.channels.types import InboundMessage, ImageRef


def _make_dispatcher(adapter, permission, captured):
    from opsrag.channels.dispatcher import ChannelDispatcher  # adapt to real name
    d = ChannelDispatcher.__new__(ChannelDispatcher)
    d._adapter = adapter
    d._permission = permission
    d._providers = type("P", (), {"embedder": None, "llm": None, "session_store": None,
                                   "vision_llm": "VISION"})()
    d._graph = None
    d._qa_cache = None
    d._investigation_cache = None
    d._semantic_router = None
    d._thread_cap = 10
    d._heartbeat_interval_s = 999
    return d


def test_denied_message_never_fetches_image():
    fetched = []

    class _Adapter:
        name = "telegram"
        async def fetch_image(self, ref): fetched.append(ref); return b"X"
        async def send_denial(self, msg, reason): return None

    class _Perm:
        async def allow(self, msg): return (False, "not allowed")

    d = _make_dispatcher(_Adapter(), _Perm(), {})
    msg = InboundMessage("c", "u", "look", "m", None, True, None,
                         images=(ImageRef(file_id="F", mime_type="image/png"),))
    asyncio.run(d.on_message(msg))
    assert fetched == []          # FR-007: no fetch on denial
```

> This test focuses on the deny path (no graph needed). The allow path is verified manually + by the ephemerality integration test (Task 16). If `on_message` references collaborators not set here on the allow path, this deny-path test still returns before reaching them.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/channels/test_dispatcher_images.py -v`
Expected: FAIL — until the fetch logic exists the test may pass vacuously OR error on import; confirm it FAILS for the right reason after Step 3 by also asserting the allow-path wiring compiles. (If it passes vacuously now, that's acceptable — Step 3 adds the real behavior the allow path needs.)

- [ ] **Step 3: Add fetch-after-permission + pass images**

In `opsrag/channels/dispatcher.py`, after the query-extract block (line 172-177) but the fetch must come AFTER permission (which is already at 156). Insert image resolution after identity resolution / before the agent call (after line 224, before line 243). Add:

```python
        # ---- Resolve inbound images (only now, post-permission) ----------
        turn_images = []
        if msg.images:
            from opsrag.llms.content import ImagePart
            for ref in msg.images:
                try:
                    raw = await self._adapter.fetch_image(ref)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("image fetch failed: %s", exc)
                    raw = None
                if raw:
                    turn_images.append(
                        ImagePart(data=raw, mime_type=ref.mime_type, name=ref.file_id or "image")
                    )
        # A bare image (no caption) still gets analyzed (spec FR-006).
        if turn_images and not user_query:
            user_query = "Please analyze this image."
            combined_query = (
                f"{thread_context}\n\n{user_query}" if thread_context else user_query
            )
```

> Move the empty-query guard: currently line 172-177 returns when `user_query` is empty. Change that guard so it only returns when there is NEITHER text NOR images:

```python
        user_query = (msg.text or "").strip()
        if not user_query and not msg.images:
            _log.info("empty query channel=%s", channel)
            return
```

Pass images + vision_llm into the agent call (lines 243-256), add:

```python
                images=turn_images,
                vision_llm=getattr(self._providers, "vision_llm", None),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/channels/test_dispatcher_images.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add opsrag/channels/dispatcher.py tests/channels/test_dispatcher_images.py
git commit -m "feat(vision): dispatcher fetches images after permission + threads to agent"
```

---

## Task 11: Telegram adapter — extract `ImageRef` + `fetch_image`

**Files:**
- Modify: `opsrag/channels/adapters/telegram/adapter.py:273-317` (`_message_to_inbound`) + add `fetch_image`
- Test: `tests/channels/test_telegram_images.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/channels/test_telegram_images.py
import asyncio
from opsrag.channels.adapters.telegram.adapter import TelegramAdapter


def _adapter():
    a = TelegramAdapter.__new__(TelegramAdapter)
    a._token = "TKN"
    return a


def test_photo_message_yields_image_ref():
    a = _adapter()
    message = {
        "message_id": 5, "chat": {"id": 7, "type": "private"},
        "from": {"id": 9}, "caption": "see this",
        "photo": [
            {"file_id": "small", "file_size": 100, "width": 90, "height": 90},
            {"file_id": "big", "file_size": 9000, "width": 1280, "height": 1280},
        ],
    }
    inbound = a._message_to_inbound(message)
    assert inbound.text == "see this"
    assert len(inbound.images) == 1
    assert inbound.images[0].file_id == "big"   # largest PhotoSize
    assert inbound.images[0].mime_type == "image/jpeg"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/channels/test_telegram_images.py -v`
Expected: FAIL — `inbound.images` is empty (photos currently ignored).

- [ ] **Step 3: Extract photos in `_message_to_inbound` + implement `fetch_image`**

In `opsrag/channels/adapters/telegram/adapter.py`, import the ref type at top:

```python
from opsrag.channels.types import ImageRef
```

In `_message_to_inbound`, before building `InboundMessage`, compute image refs:

```python
        image_refs: list[ImageRef] = []
        photos = message.get("photo") or []
        if photos:
            largest = max(photos, key=lambda p: p.get("file_size", 0) or 0)
            image_refs.append(ImageRef(
                file_id=largest.get("file_id"),
                mime_type="image/jpeg",   # Telegram photos are always JPEG
                size=largest.get("file_size"),
            ))
        doc = message.get("document") or {}
        if doc and str(doc.get("mime_type", "")).startswith("image/"):
            image_refs.append(ImageRef(
                file_id=doc.get("file_id"),
                mime_type=doc.get("mime_type"),
                size=doc.get("file_size"),
            ))
```

Add `images=tuple(image_refs),` to the `InboundMessage(...)` constructor.

Add the `fetch_image` method to the adapter class:

```python
    async def fetch_image(self, ref) -> bytes | None:
        """Telegram two-step: getFile(file_id) -> download file_path."""
        import httpx
        base = f"https://api.telegram.org/bot{self._token}"
        async with httpx.AsyncClient(timeout=30) as client:
            meta = await client.get(f"{base}/getFile", params={"file_id": ref.file_id})
            meta.raise_for_status()
            file_path = meta.json()["result"]["file_path"]
            dl = await client.get(
                f"https://api.telegram.org/file/bot{self._token}/{file_path}"
            )
            dl.raise_for_status()
            return dl.content
```

> Use the adapter's existing token attribute name. If it's `self._bot_token` (not `self._token`), use that in BOTH the test and the method.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/channels/test_telegram_images.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add opsrag/channels/adapters/telegram/adapter.py tests/channels/test_telegram_images.py
git commit -m "feat(vision): telegram extracts photo/document ImageRef + fetch_image"
```

---

## Task 12: Discord adapter — extract `ImageRef` + `fetch_image`

**Files:**
- Modify: `opsrag/channels/adapters/discord/adapter.py:464-519` (`_message_to_inbound`) + add `fetch_image`
- Test: `tests/channels/test_discord_images.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/channels/test_discord_images.py
from opsrag.channels.adapters.discord.adapter import DiscordAdapter


class _Att:
    def __init__(self, url, ct): self.url = url; self.content_type = ct; self.size = 12; self.filename = "a.png"


class _Msg:
    content = "look <@123>"
    id = 55
    attachments = [_Att("https://cdn/a.png", "image/png"), _Att("https://cdn/x.txt", "text/plain")]
    class author: id = 9; bot = False
    class channel: id = 7
    guild = None


def test_discord_extracts_only_image_attachments():
    a = DiscordAdapter.__new__(DiscordAdapter)
    inbound = a._message_to_inbound(_Msg())
    assert len(inbound.images) == 1
    assert inbound.images[0].url == "https://cdn/a.png"
    assert inbound.images[0].mime_type == "image/png"
```

> If `_message_to_inbound` needs more attributes on the fake message (e.g. `webhook_id`, `type`), add them to `_Msg` to satisfy the real parsing path — match what the method actually reads.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/channels/test_discord_images.py -v`
Expected: FAIL — `inbound.images` empty.

- [ ] **Step 3: Extract attachments + `fetch_image`**

In `opsrag/channels/adapters/discord/adapter.py`, import `ImageRef`. In `_message_to_inbound`, before building `InboundMessage`:

```python
        image_refs = tuple(
            ImageRef(url=att.url, mime_type=att.content_type, size=getattr(att, "size", None))
            for att in (getattr(message, "attachments", None) or [])
            if str(getattr(att, "content_type", "") or "").startswith("image/")
        )
```

Add `images=image_refs,` to the `InboundMessage(...)` constructor.

Add `fetch_image` (Discord CDN urls need no auth):

```python
    async def fetch_image(self, ref) -> bytes | None:
        import httpx
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(ref.url)
            resp.raise_for_status()
            return resp.content
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/channels/test_discord_images.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add opsrag/channels/adapters/discord/adapter.py tests/channels/test_discord_images.py
git commit -m "feat(vision): discord extracts image attachments + fetch_image"
```

---

## Task 13: Slack adapter — extract `ImageRef` + `fetch_image` (bearer auth)

**Files:**
- Modify: `opsrag/channels/adapters/slack/adapter.py:317-335` (`_event_to_inbound`) + add `fetch_image`
- Test: `tests/channels/test_slack_images.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/channels/test_slack_images.py
from opsrag.channels.adapters.slack.adapter import SlackAdapter


def test_slack_extracts_image_files():
    a = SlackAdapter.__new__(SlackAdapter)
    event = {
        "channel": "C1", "user": "U1", "text": "hi", "ts": "1.1", "team": "T1",
        "files": [
            {"url_private": "https://files.slack/a.png", "mimetype": "image/png", "size": 10},
            {"url_private": "https://files.slack/b.pdf", "mimetype": "application/pdf"},
        ],
    }
    inbound = a._event_to_inbound(event, is_dm=True)
    assert len(inbound.images) == 1
    assert inbound.images[0].url == "https://files.slack/a.png"
```

> Match `_event_to_inbound`'s real signature (it may take only `event`, or `(event, is_dm)`). Adjust the call accordingly.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/channels/test_slack_images.py -v`
Expected: FAIL — `inbound.images` empty.

- [ ] **Step 3: Extract files + `fetch_image` with bearer token**

In `opsrag/channels/adapters/slack/adapter.py`, import `ImageRef`. In `_event_to_inbound`, before building `InboundMessage`:

```python
        image_refs = tuple(
            ImageRef(url=f.get("url_private"), mime_type=f.get("mimetype", "image/png"),
                     size=f.get("size"))
            for f in ((event or {}).get("files") or [])
            if str(f.get("mimetype", "") or "").startswith("image/") and f.get("url_private")
        )
```

Add `images=image_refs,` to the `InboundMessage(...)` constructor.

Add `fetch_image` (Slack `url_private` requires the bot token):

```python
    async def fetch_image(self, ref) -> bytes | None:
        import httpx
        headers = {"Authorization": f"Bearer {self._bot_token}"}
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(ref.url, headers=headers)
            resp.raise_for_status()
            return resp.content
```

> Use the adapter's real bot-token attribute name (search the file for the token used in Slack Web API calls).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/channels/test_slack_images.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add opsrag/channels/adapters/slack/adapter.py tests/channels/test_slack_images.py
git commit -m "feat(vision): slack extracts image files + authed fetch_image"
```

---

## Task 14: Teams adapter — extract `ImageRef` + `fetch_image`

**Files:**
- Modify: `opsrag/channels/adapters/teams/router.py:120-156` (`activity_to_inbound`); add `fetch_image` to the Teams adapter class.
- Test: `tests/channels/test_teams_images.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/channels/test_teams_images.py
from opsrag.channels.adapters.teams.router import activity_to_inbound


def test_teams_extracts_image_attachments():
    activity = {
        "id": "a1", "text": "look",
        "from": {"id": "u1"},
        "conversation": {"id": "c1", "conversationType": "personal"},
        "attachments": [
            {"contentType": "image/png", "contentUrl": "https://teams/a.png", "name": "a.png"},
            {"contentType": "text/html", "content": "<p>hi</p>"},
        ],
    }
    inbound = activity_to_inbound(activity)
    assert len(inbound.images) == 1
    assert inbound.images[0].url == "https://teams/a.png"
    assert inbound.images[0].mime_type == "image/png"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/channels/test_teams_images.py -v`
Expected: FAIL — `inbound.images` empty.

- [ ] **Step 3: Extract attachments + `fetch_image`**

In `opsrag/channels/adapters/teams/router.py`, import `ImageRef`. In `activity_to_inbound`, before building `InboundMessage`:

```python
    image_refs = tuple(
        ImageRef(url=att.get("contentUrl"), mime_type=att.get("contentType", "image/png"))
        for att in (activity.get("attachments") or [])
        if str(att.get("contentType", "") or "").startswith("image/") and att.get("contentUrl")
    )
```

Add `images=image_refs,` to the `InboundMessage(...)` constructor.

Add `fetch_image` to the Teams adapter class (Teams-hosted content needs the bot's bearer token; public `contentUrl`s don't — try authed, fall back to anonymous):

```python
    async def fetch_image(self, ref) -> bytes | None:
        import httpx
        headers = {}
        token = await self._bot_access_token()  # existing bot-token helper
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(ref.url, headers=headers)
            if resp.status_code == 401 and headers:
                resp = await client.get(ref.url)   # retry anonymous
            resp.raise_for_status()
            return resp.content
```

> If the Teams adapter has no existing bot-token helper, fetch anonymously (drop the `Authorization` header). Wire `fetch_image` onto whichever class implements the `ChannelAdapter` Protocol for Teams.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/channels/test_teams_images.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add opsrag/channels/adapters/teams/router.py tests/channels/test_teams_images.py
git commit -m "feat(vision): teams extracts image attachments + fetch_image"
```

---

## Task 15: Web UI — attach / paste / drag-drop + send + render

**Files:**
- Modify: `ui/src/components/ChatInput.tsx`
- Modify: `ui/src/api.ts` (`streamQuery`, ~560-599)
- Modify: `ui/src/components/ChatMessage.tsx` (`Message` interface ~19-43 + render)
- Test: manual (UI), plus a `tsc` build check.

- [ ] **Step 1: Add image state + controls to `ChatInput.tsx`**

Extend the component to hold pending images as `{ mime: string; dataUrl: string; b64: string; name: string }[]`. Add (a) a hidden `<input type="file" accept="image/*" multiple>` triggered by a 📎 button, (b) an `onPaste` handler on the textarea reading `e.clipboardData.files`, (c) `onDragOver`/`onDrop` on the input container. Each selected file is read via `FileReader.readAsDataURL`; split the data URL into mime + base64. Render thumbnail chips with an ✕ remove. On send, pass the images up to the parent's submit handler alongside the text, then clear them.

```tsx
// helper (top of ChatInput.tsx)
type PendingImage = { mime: string; dataUrl: string; b64: string; name: string };

function readFileAsImage(file: File): Promise<PendingImage> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => {
      const dataUrl = String(r.result);
      const [, mime = "image/png", b64 = ""] =
        /^data:(.*?);base64,(.*)$/.exec(dataUrl) || [];
      resolve({ mime, dataUrl, b64, name: file.name });
    };
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}
```

Maintain `const [images, setImages] = useState<PendingImage[]>([])`, an `addFiles(files: FileList | File[])` that filters `image/*` and appends, and clear on submit. The submit handler calls the existing send callback with `{ text: value, images }` (extend its prop type to accept images).

- [ ] **Step 2: Send images from `api.ts`**

Change `streamQuery` to accept images and include them in the body:

```ts
export async function* streamQuery(
  query: string,
  userId: string,
  threadId?: string,
  images?: { mime_type: string; data: string }[],
) {
  // ... existing setup ...
  body: JSON.stringify({
    query,
    user_id: userId,
    thread_id: threadId ?? null,
    stream: true,
    images: images && images.length ? images : null,
  }),
  // ... existing stream handling ...
}
```

Map the UI `PendingImage[]` to `{ mime_type, data }[]` at the call site (`App.tsx` or wherever `streamQuery` is invoked): `images.map(i => ({ mime_type: i.mime, data: i.b64 }))`.

- [ ] **Step 3: Render the user's thumbnails in `ChatMessage.tsx`**

Add to the `Message` interface (after `ts?`):

```ts
  images?: { mime: string; dataUrl: string }[];
```

In the user-message render branch, if `message.images?.length`, render a small flex row of `<img>` thumbnails (max-height ~120px, rounded). These are client-only (the server is ephemeral) — when the user sends, push the `dataUrl`s onto the optimistic user `Message` so they appear in the transcript for the session.

- [ ] **Step 4: Build check**

Run: `cd ui && npm run build`
Expected: `tsc` + Vite build succeed with no type errors.

- [ ] **Step 5: Commit**

```bash
git add ui/src/components/ChatInput.tsx ui/src/api.ts ui/src/components/ChatMessage.tsx
git commit -m "feat(vision): UI image attach/paste/drag-drop + send + thumbnail render"
```

---

## Task 16: Ephemerality integration test (SC-002)

**Files:**
- Test: `tests/agent/test_vision_ephemeral.py` (append)

- [ ] **Step 1: Write the test asserting no bytes reach the persisted layer**

```python
# tests/agent/test_vision_ephemeral.py  (append)
def test_marker_in_state_no_bytes():
    """The state/checkpoint sees a text marker; image bytes never appear."""
    from opsrag.agent.graph import query_with_session_events
    g = _FakeGraph()

    async def _run():
        async for _ in query_with_session_events(
            g, query="diagnose", user_id="u", thread_id="web:u:t",
            images=[ImagePart(b"SECRETIMAGEBYTES", "image/png", "err.png")],
            vision_llm=object(),
        ):
            pass

    asyncio.run(_run())
    blob = repr(g.seen_initial)
    assert "SECRETIMAGEBYTES" not in blob
    assert "[attached image: err.png]" in g.seen_initial["query"]
```

- [ ] **Step 2: Run it**

Run: `pytest tests/agent/test_vision_ephemeral.py -v`
Expected: PASS (both tests in the file).

- [ ] **Step 3: Run the full new suite**

Run: `pytest tests/llms/test_content.py tests/llms/test_provider_multimodal.py tests/test_config_vision.py tests/test_factory_vision.py tests/agent/test_generator_vision.py tests/agent/test_vision_ephemeral.py tests/api/test_query_images.py tests/channels/test_types_images.py tests/channels/test_dispatcher_images.py tests/channels/test_telegram_images.py tests/channels/test_discord_images.py tests/channels/test_slack_images.py tests/channels/test_teams_images.py -v`
Expected: ALL PASS.

- [ ] **Step 4: Run the existing suite for regressions**

Run: `pytest tests/llms tests/agent tests/channels tests/api -q`
Expected: no regressions (text-only fast path is byte-identical).

- [ ] **Step 5: Commit**

```bash
git add tests/agent/test_vision_ephemeral.py
git commit -m "test(vision): assert image bytes never reach the checkpoint (SC-002)"
```

---

## Task 17: Docs + sample config

**Files:**
- Modify: `README.md` (capabilities + a vision config snippet)
- Modify: the example config / Helm values that document `llm:` (add a `vision:` block)
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add a `vision:` config example**

Document the YAML + env overrides (no rebuild):

```yaml
vision:
  enabled: true
  # Auto-route fallback used only when the active model can't see.
  # AWS/Bedrock: claude-sonnet-4-6 ; GCP/Vertex: gemini-3-flash-preview
  model: null          # or e.g. "anthropic.claude-sonnet-4-6-v1:0"
  provider: null       # defaults to the main llm.provider
  max_images: 4
  max_bytes: 5242880   # 5 MB
  allowed_mime: ["image/png", "image/jpeg", "image/gif", "image/webp"]
```

Env equivalents: `OPSRAG_VISION_MODEL`, `OPSRAG_VISION_ENABLED`, etc. (wire any new env reads in `config.py`'s loader if it uses an env-override pass — match the existing pattern, e.g. how `OPSRAG_LLM_MODEL` is read in `factory.py`).

- [ ] **Step 2: Update README capabilities + CHANGELOG**

State that OpsRAG now understands attached images on web + Telegram/Discord/Slack/Teams, ephemerally, with provider-aware vision routing.

- [ ] **Step 3: Commit**

```bash
git add README.md CHANGELOG.md <example-config-or-values-file>
git commit -m "docs(vision): document image/vision support + vision config block"
```

---

## Self-Review

**Spec coverage:**
- FR-001 (attach on web + all channels) → Tasks 8, 11–15
- FR-002 (pass-through to vision LLM) → Tasks 4, 7
- FR-003 (ephemeral, no checkpoint bytes) → Tasks 6, 16 (asserted)
- FR-004 (auto-route to vision model) → Tasks 5, 7
- FR-005 (no vision → drop + notice) → Task 7
- FR-006 (bare image → auto-analyze) → Task 10
- FR-007 (fetch only after permission) → Task 10 (asserted)
- FR-008 (authz unchanged) → Task 10 reuses existing permission gate; no authz code touched
- FR-009 (count + size caps) → Tasks 3, 8 (web) + 8's `decode_images`; channel caps enforced by reusing `VisionConfig` — **add a cap check in the dispatcher** (see note below)
- FR-010 (mime allow-list) → Tasks 3, 8; channel adapters pre-filter to `image/*` (Tasks 11–14)
- FR-011 (configurable, provider-aware default) → Tasks 2, 3, 5
- FR-012 (usage telemetry) → no change needed; every provider's `generate()` already calls `tracker.record` (verified in provider files) and vision tokens flow through it
- FR-013 (reject whole turn on invalid) → Task 8 `decode_images` raises on first invalid (web). Channels drop unfetchable images individually (Task 10) — acceptable since channel platforms pre-validate uploads.
- FR-014 (failed fetch → text-only, no crash) → Task 10 (try/except per ref)

**Gap found + fix:** FR-009 caps must also apply to channel images. Add to Task 10, Step 3, after building `turn_images`:

```python
        if len(turn_images) > config_max_images:   # read from VisionConfig
            turn_images = turn_images[:config_max_images]
            _log.info("clamped channel images to max")
```

Thread the limits into the dispatcher (it already receives providers/config at construction — pass `vision_config` or read `max_images`/`max_bytes` and enforce per-ref `len(raw) <= max_bytes` in the fetch loop). If the dispatcher has no config handle, add a `vision: VisionConfig` constructor arg in the dispatcher's `__init__` and pass it from the channels boot wiring.

**Placeholder scan:** none — every code step shows real code; the few "adapt to the real attribute name" notes are disambiguation guidance, not missing logic.

**Type consistency:** `ImagePart(data, mime_type, name)`, `ImageRef(file_id, url, mime_type, size)`, `build_user_content(text, images)`, `to_*_content(content)`, `is_vision_capable(provider, model)`, `default_vision_model(provider)`, `resolve_vision_model(vision, llm_cfg)`, `decode_images(images, vision)` — names used identically across all tasks. `turn_images` / `vision_llm` are the config keys everywhere (Tasks 6, 7, 10).
