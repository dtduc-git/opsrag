"""Confidence + cc rendering for first-responder replies.

Wraps ``format_answer_as_slack_blocks`` (unchanged) and adds two things the
shared render deliberately does not carry:

  * a confidence CONTEXT block prepended above the answer, derived from the
    final event's ``grounded`` flag + source count;
  * an on-call cc as a standalone SECTION block AND in the notification text.

The cc is NEVER concatenated into the answer prose: the answer body is
truncated at 2800 chars (render.py) and the fallback strips ``>`` -- either
would corrupt/drop the ``<!subteam^...>`` ping. A standalone block is
structurally immune to the answer-body truncation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from opsrag.slack_bot.render import format_answer_as_slack_blocks

_BARE_SUBTEAM_RE = re.compile(r"^S[A-Z0-9]{6,}$")


@dataclass(frozen=True)
class Confidence:
    """A rendered confidence verdict."""

    label: str  # "high" | "medium" | "low"
    emoji: str
    note: str


def derive_confidence(final: dict[str, Any]) -> Confidence:
    """Map a final agent event to a confidence verdict.

    The ONLY grounding signal on the final event is ``grounded`` (bool).
    high  = grounded AND >= 1 source
    medium= grounded AND 0 sources
    low   = not grounded (includes missing field)
    """
    grounded = bool((final or {}).get("grounded"))
    n_sources = len((final or {}).get("sources") or [])
    if grounded and n_sources >= 1:
        return Confidence("high", "🟢", f"grounded in {n_sources} source(s)")
    if grounded:
        return Confidence("medium", "🟡", "grounded, but no cited sources")
    return Confidence(
        "low", "🔴", "⚠️ unverified — please confirm before acting",
    )


def normalize_oncall_handle(handle: str) -> str:
    """Coerce a bare ``S...`` subteam id to a ``<!subteam^S...>`` mention.

    A bare id or plain text does NOT ping; only the mention token does.
    Already-formed tokens and empty strings pass through unchanged.
    """
    h = (handle or "").strip()
    if not h:
        return ""
    if _BARE_SUBTEAM_RE.match(h):
        return f"<!subteam^{h}>"
    return h


def _confidence_block(c: Confidence) -> dict[str, Any]:
    return {
        "type": "context",
        "elements": [
            {"type": "mrkdwn", "text": f"{c.emoji} *{c.label.upper()} confidence* · {c.note}"},
        ],
    }


def _cc_block(cc_mention: str) -> dict[str, Any]:
    return {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"cc {cc_mention} — routing to on-call for awareness.",
        },
    }


def build_reply(
    *,
    answer: str,
    sources: list[dict[str, Any]],
    confidence: Confidence,
    oncall_handle: str,
    diagram_present: bool = False,
    web_ui_base_url: str = "",
    session_id: str | None = None,
    investigation_id: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Build ``(text, blocks)`` for a first-responder reply.

    Layout: [confidence context] + [answer/sources/feedback/footer blocks] +
    [cc section]. The cc mention is also appended to ``text`` so the subteam
    ping fires even on clients that don't render blocks.
    """
    fallback_text, answer_blocks = format_answer_as_slack_blocks(
        answer,
        sources,
        diagram_present=diagram_present,
        web_ui_base_url=web_ui_base_url,
        session_id=session_id,
        investigation_id=investigation_id,
    )
    cc = normalize_oncall_handle(oncall_handle)
    blocks: list[dict[str, Any]] = [_confidence_block(confidence), *answer_blocks]
    text = fallback_text
    if cc:
        blocks.append(_cc_block(cc))
        text = f"{fallback_text}\ncc {cc}"
    return text, blocks
