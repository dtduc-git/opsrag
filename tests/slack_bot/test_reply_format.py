"""Confidence derivation + cc/reply rendering tests."""
from __future__ import annotations

from opsrag.slack_bot.reply_format import (
    build_reply,
    derive_confidence,
    normalize_oncall_handle,
)

CC = "<!subteam^S0EXAMPLE01>"


def test_confidence_high_when_grounded_with_sources():
    c = derive_confidence({"grounded": True, "sources": [{"title": "x"}]})
    assert c.label == "high"


def test_confidence_medium_when_grounded_no_sources():
    c = derive_confidence({"grounded": True, "sources": []})
    assert c.label == "medium"


def test_confidence_low_when_not_grounded():
    c = derive_confidence({"grounded": False, "sources": [{"title": "x"}]})
    assert c.label == "low"


def test_confidence_low_when_field_absent():
    c = derive_confidence({})
    assert c.label == "low"


def test_normalize_bare_subteam_id():
    assert normalize_oncall_handle("S0EXAMPLE01") == CC


def test_normalize_already_formed_token_is_unchanged():
    assert normalize_oncall_handle(CC) == CC


def test_normalize_empty_is_empty():
    assert normalize_oncall_handle("") == ""


def test_build_reply_prepends_confidence_and_appends_cc_block():
    text, blocks = build_reply(
        answer="Restart the deployment.",
        sources=[{"title": "runbook", "url": "https://x"}],
        confidence=derive_confidence({"grounded": True, "sources": [{"title": "r"}]}),
        oncall_handle="S0EXAMPLE01",
        diagram_present=False,
        web_ui_base_url="",
        session_id=None,
        investigation_id=None,
    )
    # cc is in the notification text (the ping surface).
    assert CC in text
    # First block is the confidence context block.
    assert blocks[0]["type"] == "context"
    assert "confidence" in blocks[0]["elements"][0]["text"].lower()
    # A standalone section block carries the cc mention (never inside prose).
    cc_blocks = [
        b for b in blocks
        if b.get("type") == "section" and CC in b.get("text", {}).get("text", "")
    ]
    assert len(cc_blocks) == 1


def test_build_reply_cc_survives_long_answer_truncation():
    long_answer = "A" * 6000  # exceeds the 2800-char answer-body cap
    text, blocks = build_reply(
        answer=long_answer,
        sources=[],
        confidence=derive_confidence({"grounded": False, "sources": []}),
        oncall_handle=CC,
        diagram_present=False,
        web_ui_base_url="",
        session_id=None,
        investigation_id=None,
    )
    # The cc must still be present in a standalone block AND the text, even
    # though the answer body was truncated.
    assert CC in text
    assert any(
        b.get("type") == "section" and CC in b.get("text", {}).get("text", "")
        for b in blocks
    )


def test_build_reply_low_confidence_note_warns():
    _, blocks = build_reply(
        answer="Maybe try X.",
        sources=[],
        confidence=derive_confidence({"grounded": False, "sources": []}),
        oncall_handle=CC,
        diagram_present=False,
        web_ui_base_url="",
        session_id=None,
        investigation_id=None,
    )
    assert "⚠️" in blocks[0]["elements"][0]["text"]


def _feedback_row(blocks):
    return next((b for b in blocks if b.get("block_id") == "opsrag_feedback_row"), None)


def test_build_reply_feedback_anchors_on_investigation_id_when_present():
    _, blocks = build_reply(
        answer="Restart the deployment.",
        sources=[],
        confidence=derive_confidence({"grounded": True, "sources": [{"title": "r"}]}),
        oncall_handle=CC,
        session_id="slack-thread:C1:1.1",
        investigation_id="inv-42",
    )
    row = _feedback_row(blocks)
    assert row is not None
    assert row["elements"][0]["value"] == "up:inv-42"
    assert row["elements"][1]["value"] == "down:inv-42"


def test_build_reply_feedback_falls_back_to_session_when_no_investigation():
    # LOW-confidence / unverified answers are NOT cached -> no investigation_id.
    # The 👍/👎 row must still render, anchored on the session/thread id
    # (colon-y id must survive the value verbatim; parser splits on first ':').
    _, blocks = build_reply(
        answer="Unverified answer -- confirm before acting.",
        sources=[],
        confidence=derive_confidence({"grounded": False, "sources": []}),
        oncall_handle=CC,
        session_id="slack-thread:C0EXAMPLE02:1784003.99",
        investigation_id=None,
    )
    row = _feedback_row(blocks)
    assert row is not None
    assert row["elements"][0]["value"] == "up:slack-thread:C0EXAMPLE02:1784003.99"
    assert row["elements"][1]["value"] == "down:slack-thread:C0EXAMPLE02:1784003.99"


def test_build_reply_no_feedback_row_without_any_anchor():
    _, blocks = build_reply(
        answer="x",
        sources=[],
        confidence=derive_confidence({"grounded": False, "sources": []}),
        oncall_handle=CC,
        session_id=None,
        investigation_id=None,
    )
    assert _feedback_row(blocks) is None
