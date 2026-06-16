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


# ---- vision capability map + provider-aware default model (Task 2) ----

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
# `vision.model` is configured (spec FR-011). Aligned with the project's
# reason/pro tier (Sonnet 4.6 -- see model_bundles.py + pricing.py); override
# with OPSRAG_VISION_MODEL / vision.model for a different id. The Bedrock id is
# the region-prefixed inference profile AWS deployments use (model_bundles "aws").
_DEFAULT_VISION_MODEL = {
    "anthropic": "claude-sonnet-4-6",
    "bedrock": "us.anthropic.claude-sonnet-4-6",
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
