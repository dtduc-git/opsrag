"""Pure normalizer tests for opsrag.slack_bot.slack_text.

Encodes the full 45-row matrix from
docs/superpowers/plans/2026-07-13-opsrag-slack-fr-phase1-polish.md §3
(Slack mrkdwn + Slack tokens -> clean markdown for the ingested query text),
plus the ``slack_ids`` id-scan assertion. Mirrors the plain-function style of
``tests/slack_bot/test_request_extract.py``.
"""
from __future__ import annotations

import pytest

from opsrag.slack_bot.slack_text import normalize_slack_text, slack_ids

USER_NAMES = {"U0EXAMPLE01": "duc", "U1": "alice", "U2": "bob"}
SUBTEAM_NAMES = {"S0EXAMPLE01": "sre-oncall"}
CHANNEL_NAMES = {"C123": "ops"}


def _norm(text: str) -> str:
    return normalize_slack_text(
        text,
        user_names=USER_NAMES,
        subteam_names=SUBTEAM_NAMES,
        channel_names=CHANNEL_NAMES,
    )


# (row #, input, expected) -- row numbers match the plan's §3 matrix table.
MATRIX: list[tuple[int, str, str]] = [
    (1, "<@U0EXAMPLE01>", "@duc"),
    (2, "<@U1> and <@U2>", "@alice and @bob"),
    (3, "<@U9NOPE>", "@U9NOPE"),
    (4, "<@U1|alice_l>", "@alice_l"),
    (5, "<!subteam^S0EXAMPLE01>", "@sre-oncall"),
    (6, "<!subteam^SZZZZZZ>", "@oncall"),
    (7, "<!subteam^S1|oncall-team>", "@oncall-team"),
    (8, "<#C123|general>", "#general"),
    (9, "<#C123>", "#ops"),
    (10, "<#C999>", "#C999"),
    (11, "<https://ex.com|Dashboard>", "[Dashboard](https://ex.com)"),
    (12, "see <https://ex.com>", "see https://ex.com"),
    (13, "<mailto:a@b.com|Email Ops>", "[Email Ops](mailto:a@b.com)"),
    (14, "<mailto:a@b.com>", "a@b.com"),
    (15, "*deploy done*", "**deploy done**"),
    (16, "~old value~", "~~old value~~"),
    (17, "_stays italic_", "_stays italic_"),
    (18, ":fire: incident", "\U0001f525 incident"),
    (19, ":firef:", "\U0001f525"),
    (20, "deploy :sparkle_custom: done", "deploy done"),
    (21, "Tom &amp; Jerry &lt;3 &gt;9000", "Tom & Jerry <3 >9000"),
    (22, "&gt; quoted line", "> quoted line"),
    (23, "&gt;&gt;&gt; big quote", "> big quote"),
    (24, "<!here> deploy now", "@here deploy now"),
    (25, "<!channel>", "@channel"),
    (26, "posted <!date^1392734382^{date}|Feb 18, 2014> ok", "posted Feb 18, 2014 ok"),
    (27, "• item one\n• item two", "- item one\n- item two"),
    (28, "use `<@U1>` literally", "use `<@U1>` literally"),
    (29, "```\n<@U1> *x* :fire:\n```", "```\n<@U1> *x* :fire:\n```"),
    (30, "<https://x.io|a &gt; b>", "[a > b](https://x.io)"),
    (31, "if a &lt; b then stop", "if a < b then stop"),
    (32, "*<@U1>* :rocket:", "**@alice** \U0001f680"),
    (33, "", ""),
    (
        34,
        "*Incident* <!subteam^S0EXAMPLE01> check <https://dash|dashboard> :fire:",
        "**Incident** @sre-oncall check [dashboard](https://dash) \U0001f525",
    ),
    # D1: bare <tel:> autolink leak fix.
    (35, "call <tel:+15551234567>", "call +15551234567"),
    (36, "<tel:+15551234567|+1 555 123 4567>", "[+1 555 123 4567](tel:+15551234567)"),
    # D2: unterminated fence must stay fully protected.
    (37, "```\n<@U1> *bold* :fire:", "```\n<@U1> *bold* :fire:"),
    # Double-unescape guard: lt/gt run before amp, so &amp;lt; decodes only
    # one layer (to literal "&lt;"), never all the way to "<".
    (38, "code &amp;lt; tag", "code &lt; tag"),
    (39, "AT&amp;T outage", "AT&T outage"),
    (40, ":fire::rocket: go", "\U0001f525\U0001f680 go"),
    # Word-glued shortcode is NOT rendered (pinned tradeoff).
    (41, "deploy done:tada:", "deploy done:tada:"),
    # mailto label == email -> dedupe to bare email.
    (42, "<mailto:a@b.com|a@b.com>", "a@b.com"),
    # Inline-code scoping: protected token untouched, live token rewritten.
    (43, "`<@U1>` and <@U2>", "`<@U1>` and @bob"),
    # Link-then-bold nesting.
    (44, "*<https://x|y>*", "**[y](https://x)**"),
    # Line-start blockquote (pinned: fixed "> " replacement adds a space).
    (45, "&gt;9000 rps", "> 9000 rps"),
]


@pytest.mark.parametrize(
    "row,text,expected", MATRIX, ids=[f"row{r:02d}" for r, _, _ in MATRIX]
)
def test_normalize_matrix(row: int, text: str, expected: str) -> None:
    assert _norm(text) == expected


def test_normalize_matrix_covers_45_rows() -> None:
    assert len(MATRIX) == 45


def test_slack_ids_scans_user_subteam_channel() -> None:
    assert slack_ids("<@U1> <@U2> <!subteam^S9> <#C7|x> plain") == (
        {"U1", "U2"},
        {"S9"},
        {"C7"},
    )


def test_slack_ids_empty_text_returns_empty_sets() -> None:
    assert slack_ids("") == (set(), set(), set())


def test_normalize_empty_text_returns_empty_string() -> None:
    assert normalize_slack_text("") == ""
    assert normalize_slack_text(None) == ""  # falsy input degrades gracefully
