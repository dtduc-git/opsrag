"""Tolerant first-object JSON extraction for LLM structured outputs.

Every provider's `generate_structured` used the same strict
`json.loads(text)` on the raw completion. LLMs in json mode occasionally
append garbage after a valid object -- captured live from
gemini-3-flash-preview 2026-07-13: a stray trailing brace
('{\\n  "grounded": false\\n}\\n}') -- and strict parsing throws
"Extra data", so the caller's fail-closed path discards a perfectly good
verdict (spurious "not grounded" -> warning notes on every answer plus
wasted regenerate loops).

`extract_first_json_object` decodes the FIRST JSON object and ignores
anything after it. Clean input parses identically to `json.loads`, so
providers that already emit clean JSON (Claude / Bedrock / OpenAI) see
zero behavior change.
"""
from __future__ import annotations

import json
from typing import Any


def extract_first_json_object(raw: str) -> Any:
    """Parse the first JSON object in `raw`, tolerating junk around it.

    Handles: markdown fences, prose before/after the object, a second
    object or stray brace after the first. Raises ValueError when no
    object can be decoded (callers keep their fail-closed behavior).
    """
    if not raw or not raw.strip():
        raise ValueError("empty LLM output -- no JSON object to extract")
    start = raw.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in LLM output: {raw[:120]!r}")
    try:
        obj, _end = json.JSONDecoder().raw_decode(raw, start)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in LLM output: {raw[:120]!r}") from exc
    return obj
