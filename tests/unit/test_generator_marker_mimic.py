"""Unit tests: generator must never ship `[called tool]` marker mimicry.

`_flatten_tool_history` renders past tool calls as assistant lines like
`[called tool] code_grep({...})` in the generator's context. Gemini
occasionally CONTINUES that pattern instead of writing the final answer —
the user then sees a bare tool-trace dump as the whole reply (observed in
prod: answer was exactly two `[called tool] code_grep(...)` markers glued
together, nothing else; the same question answered fine minutes earlier).

Guard under test: `_strip_tool_marker_mimicry(draft)` removes mimicked
markers from a draft and flags drafts that are NOTHING BUT markers so the
generator node can retry once with a corrective note (same pattern as the
fabricated-citation guard) instead of shipping the dump.
"""
from __future__ import annotations

from opsrag.agent.nodes.multi_agent import _strip_tool_marker_mimicry

REAL_DUMP = (
    '[called tool] code_grep({"pattern": "eventbus.pg.acme-notes-be.widget-order", '
    '"repo": "saas/acme-notes-be"})'
    '[called tool] code_grep({"pattern": "widget_events", "repo": "saas/acme-notes-be"})'
)


def test_pure_marker_dump_is_flagged():
    cleaned, is_dump = _strip_tool_marker_mimicry(REAL_DUMP)
    assert is_dump is True
    assert cleaned.strip() == ""


def test_markers_with_real_answer_are_stripped_not_flagged():
    draft = (
        "The consumer is `acme-notes-appservice-consumers` in namespace "
        "acme-notes, confirmed by the KEDA scaler on the topic.\n"
        + REAL_DUMP
    )
    cleaned, is_dump = _strip_tool_marker_mimicry(draft)
    assert is_dump is False
    assert "[called tool]" not in cleaned
    assert "acme-notes-appservice-consumers" in cleaned


def test_clean_answer_unchanged():
    draft = "Deployment X consumes topic Y [prometheus_query]."
    cleaned, is_dump = _strip_tool_marker_mimicry(draft)
    assert cleaned == draft
    assert is_dump is False


def test_whitespace_separated_dump_flagged():
    draft = '[called tool] k8s_find_workloads({"name_contains": "x"})\n\n' \
            '[called tool] prometheus_query({"query": "up"})\n'
    cleaned, is_dump = _strip_tool_marker_mimicry(draft)
    assert is_dump is True


def test_tiny_residue_still_counts_as_dump():
    # A couple of stray characters around markers is not a real answer.
    draft = "Ok.\n" + REAL_DUMP
    cleaned, is_dump = _strip_tool_marker_mimicry(draft)
    assert is_dump is True
