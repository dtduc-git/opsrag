"""Render an OpsRAG answer + sources as Slack Block Kit blocks.

Slack uses ``mrkdwn``, a markdown-ish dialect with notable differences
from standard markdown -- see https://api.slack.com/reference/surfaces/formatting.
We convert the agent's standard markdown output to mrkdwn before
embedding it in a ``section`` block.

Layout (top -> bottom):

1. Answer prose (markdown -> mrkdwn)
2. (optional) diagram callout -- "Diagram available -- open in OpsRAG UI"
3. Sources list (bullets, up to 10, "+N more" overflow)
4. Footer context block -- "OpsRAG · <timestamp>" + "View full answer ->"
"""
from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any

_log = logging.getLogger("opsrag.slack_bot.render")

# Hard cap on bullet rows we render -- Slack truncates very long
# section blocks aggressively (3000 char limit per text field).
_MAX_SOURCES = 10
_TRUNCATION_SUFFIX_PLAIN = "\n\n_...answer truncated_"
_TRUNCATION_SUFFIX_WITH_LINK = "\n\n_...answer truncated -- <{url}|view full in OpsRAG UI>_"


# =====================================================================
# Markdown -> mrkdwn conversion
# =====================================================================

# Pattern fragments. Order matters -- we must protect fenced code blocks
# from inline rewrites because mrkdwn inside ``` blocks is literal.
_FENCED_CODE_RE = re.compile(r"```(.*?)```", re.DOTALL)
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_BOLD_UNDERSCORE_RE = re.compile(r"(?<![_\w])__([^_]+)__(?![_\w])")
_ITALIC_RE = re.compile(r"(?<![*\w])\*([^*\n]+?)\*(?![*\w])")
_ITALIC_UNDERSCORE_RE = re.compile(r"(?<![_\w])_([^_\n]+?)_(?![_\w])")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def _to_mrkdwn(text: str) -> str:
    """Convert a CommonMark-ish answer to Slack mrkdwn.

    Conversions:
      - fenced code blocks preserved as-is (mrkdwn supports ``` ```)
      - inline ``[text](url)`` -> ``<url|text>``
      - ``**bold**`` and ``__bold__`` -> ``*bold*``
      - ``*italic*`` and ``_italic_`` -> ``_italic_``
      - ``#`` / ``##`` / ... headings -> bolded line (Slack has no native
        heading syntax in mrkdwn)
      - markdown tables -> indented bullet list (Slack doesn't render
        tables)
    """
    if not text:
        return ""

    # 1. Carve out fenced code blocks so we don't mangle their contents.
    placeholders: list[str] = []

    def _stash_code(match: re.Match[str]) -> str:
        placeholders.append(match.group(0))
        return f"\x00CODE{len(placeholders) - 1}\x00"

    work = _FENCED_CODE_RE.sub(_stash_code, text)

    # 2. Convert markdown tables to bullet lists.
    work = _tables_to_bullets(work)

    # Stash bold-formatted spans behind placeholders so the italic
    # pass below doesn't see the single-asterisks we're about to write
    # and convert them to underscores.
    bold_placeholders: list[str] = []

    def _stash_as_bold(rendered: str) -> str:
        bold_placeholders.append(rendered)
        return f"\x00BOLD{len(bold_placeholders) - 1}\x00"

    # 3. Headings -> bold line.
    work = _HEADING_RE.sub(
        lambda m: _stash_as_bold(f"*{m.group(2).strip()}*"),
        work,
    )

    # 4. Links: [t](u) -> <u|t>
    work = _LINK_RE.sub(lambda m: f"<{m.group(2)}|{m.group(1)}>", work)

    # 5. Bold (both ** and __ flavours) -> *...*.
    work = _BOLD_RE.sub(lambda m: _stash_as_bold(f"*{m.group(1)}*"), work)
    work = _BOLD_UNDERSCORE_RE.sub(
        lambda m: _stash_as_bold(f"*{m.group(1)}*"), work
    )

    # 6. Italic -> _..._. Standard markdown uses *...* OR _..._ for italic;
    #    Slack only accepts _..._.
    work = _ITALIC_RE.sub(lambda m: f"_{m.group(1)}_", work)
    # _italic_ is already mrkdwn-compatible, no-op except for normalising
    # whitespace edges.
    work = _ITALIC_UNDERSCORE_RE.sub(lambda m: f"_{m.group(1)}_", work)

    # Restore bold spans.
    def _restore_bold(match: re.Match[str]) -> str:
        return bold_placeholders[int(match.group(1))]

    work = re.sub(r"\x00BOLD(\d+)\x00", _restore_bold, work)

    # 7. Restore protected code blocks.
    def _restore(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        return placeholders[idx]

    work = re.sub(r"\x00CODE(\d+)\x00", _restore, work)
    return work


def _tables_to_bullets(text: str) -> str:
    """Crude markdown-table -> bullet-list converter.

    Slack mrkdwn can't render pipe tables. Rather than dropping them
    we extract the header + row cells and emit ``- header1: value1,
    header2: value2`` style bullets. Imperfect but readable.
    """
    out_lines: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if (
            _TABLE_ROW_RE.match(line)
            and i + 1 < len(lines)
            and _TABLE_SEPARATOR_RE.match(lines[i + 1])
        ):
            # We have a header row + separator row + N data rows.
            header_cells = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 2  # skip header + separator
            while i < len(lines) and _TABLE_ROW_RE.match(lines[i]):
                data_cells = [
                    c.strip() for c in lines[i].strip().strip("|").split("|")
                ]
                pairs = []
                for h, v in zip(header_cells, data_cells):
                    if h and v:
                        pairs.append(f"{h}: {v}")
                    elif v:
                        pairs.append(v)
                if pairs:
                    out_lines.append("* " + ", ".join(pairs))
                i += 1
            continue
        out_lines.append(line)
        i += 1
    return "\n".join(out_lines)


# =====================================================================
# Block Kit assembly
# =====================================================================


def _truncate(text: str, cap: int, deep_link: str | None = None) -> str:
    """Clip text to ``cap`` chars and append a truncation marker.

    When ``deep_link`` is set, the marker links to the OpsRAG UI so the
    user can view the full (untrimmed) answer. Otherwise we just say
    "_answer truncated_".
    """
    if deep_link:
        suffix = _TRUNCATION_SUFFIX_WITH_LINK.format(url=deep_link)
    else:
        suffix = _TRUNCATION_SUFFIX_PLAIN
    if len(text) <= cap:
        return text
    head = text[: max(0, cap - len(suffix))]
    # Avoid splitting mid-codefence -- back off to the previous newline
    # if we're inside an obviously unclosed fence.
    if head.count("```") % 2 == 1:
        last_fence = head.rfind("```")
        if last_fence > 0:
            head = head[:last_fence].rstrip()
    return head.rstrip() + suffix


def _render_sources(sources: list[dict[str, Any]]) -> tuple[str, int]:
    """Return (bullet_list_text, total_count_or_zero).

    Empty list -> ``("", 0)``. Caller decides whether to emit a Sources
    section at all based on the count.
    """
    if not sources:
        return "", 0
    bullets: list[str] = []
    for src in sources[:_MAX_SOURCES]:
        if not isinstance(src, dict):
            continue
        title = (src.get("title") or src.get("source") or "source").strip()
        url = (src.get("url") or "").strip()
        path = (src.get("source") or "").strip()
        if url:
            bullets.append(f"* <{url}|{title}>")
        elif path:
            bullets.append(f"* `{path}`")
        else:
            bullets.append(f"* {title}")
    extra = len(sources) - _MAX_SOURCES
    if extra > 0:
        bullets.append(f"* _+{extra} more_")
    return "\n".join(bullets), len(sources)


def _deep_link(web_ui_base_url: str, session_id: str | None) -> str | None:
    base = (web_ui_base_url or "").rstrip("/")
    if not base:
        return None
    if session_id:
        return f"{base}/#chat/{session_id}"
    return base


def format_answer_as_slack_blocks(
    answer: str,
    sources: list[dict[str, Any]],
    *,
    diagram_present: bool = False,
    web_ui_base_url: str = "",
    session_id: str | None = None,
    chars_cap: int = 2800,
    investigation_id: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Render the agent's answer + sources as ``(fallback_text, blocks)``.

    Parameters
    ----------
    answer:
        Markdown answer from the agent.
    sources:
        List of source dicts, shape: ``{"source": ..., "repo": ...,
        "url": ..., "title": ...}``. URL + title are optional.
    diagram_present:
        Whether the agent produced a Mermaid/JSON diagram payload. If
        True we add a "View in OpsRAG UI" callout instead of trying to
        render the diagram inline.
    web_ui_base_url:
        Base URL of the OpsRAG web UI for deep links (e.g.
        ``https://opsrag.example.com``). If empty, no deep links are
        produced.
    session_id:
        Session id for the deep link.
    chars_cap:
        Hard cap on the answer-prose text field. Slack rejects section
        blocks with ``text`` > 3000 chars; default 2800 leaves headroom
        for the truncation suffix.

    Returns
    -------
    (fallback_text, blocks)
        ``fallback_text`` is used by Slack for mobile notifications and
        clients that don't render blocks. ``blocks`` is the Block Kit
        array ready for ``chat.update``.
    """
    fallback_text = _answer_to_fallback(answer)
    blocks: list[dict[str, Any]] = []
    # Computed once and reused for the truncation suffix + the footer.
    deep_link = _deep_link(web_ui_base_url, session_id)

    # 1. Answer body.
    body_mrkdwn = _truncate(_to_mrkdwn(answer or ""), chars_cap, deep_link=deep_link)
    if body_mrkdwn:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": body_mrkdwn},
        })

    # 2. Diagram callout.
    if diagram_present:
        diagram_text = (
            "Diagram available -- open in OpsRAG UI for the full visual."
        )
        if deep_link:
            diagram_text += f" <{deep_link}|View in OpsRAG UI>"
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": diagram_text},
        })
        if deep_link:
            blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "View diagram in UI"},
                        "url": deep_link,
                        "style": "primary",
                    }
                ],
            })

    # 3. Sources.
    sources_text, total = _render_sources(sources or [])
    if sources_text:
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Sources* ({total})\n{sources_text}",
            },
        })

    # 3.5. Inline 👍/👎 feedback. Anchor the vote on the `investigation_id`
    # when we have one (grounded tool-path answers get cached + carry a UUID
    # -> the vote resolves the exact turn + its query/answer snippets).
    # Otherwise fall back to the `session_id` (the `slack-thread:C:ts` id):
    # LOW-confidence / unverified answers are deliberately NOT cached, so they
    # have no investigation_id -- but feedback matters MOST on those, so we
    # still render the row anchored on the thread. The feedback path resolves
    # thread-shaped ids (grounded turns stored under this thread resolve +
    # carry snippets; ungrounded ones record the thumbs against the session,
    # with the answer snippet captured from the click payload). Slack posts the
    # click with `value` `<direction>:<anchor>` and `action_id`
    # `opsrag_feedback_*`; the parser splits on the FIRST ':' so a colon-y
    # session id survives verbatim. No anchor at all -> omit the row.
    feedback_anchor = investigation_id or session_id
    if feedback_anchor:
        blocks.append({
            "type": "actions",
            "block_id": "opsrag_feedback_row",
            "elements": [
                {
                    "type": "button",
                    "action_id": "opsrag_feedback_up",
                    "text": {"type": "plain_text", "text": "👍 Helpful", "emoji": True},
                    "style": "primary",
                    "value": f"up:{feedback_anchor}",
                },
                {
                    "type": "button",
                    "action_id": "opsrag_feedback_down",
                    "text": {"type": "plain_text", "text": "👎 Wrong", "emoji": True},
                    "style": "danger",
                    "value": f"down:{feedback_anchor}",
                },
            ],
        })

    # 4. Footer -- attribution + timestamp + "View in OpsRAG UI" link.
    # The link opens the same conversation in the OpsRAG web UI where
    # the user gets the full sources panel, diagram (if any), and the
    # thumbs-up/down feedback controls. In prod the base URL is
    # the deployed OpsRAG web UI URL behind Pomerium SSO; in local dev
    # it's localhost:5173 (or empty, in which case we omit the link).
    footer_text = (
        "OpsRAG · "
        + datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    )
    if deep_link:
        footer_text += f" · <{deep_link}|View in OpsRAG UI ->>"
    blocks.append({
        "type": "context",
        "elements": [
            {"type": "mrkdwn", "text": footer_text},
        ],
    })

    return fallback_text, blocks


def _answer_to_fallback(answer: str) -> str:
    """Best-effort plain-text version for mobile notifications.

    Strips markdown formatting so the notification is readable in the
    push preview. Caps to 200 chars.
    """
    if not answer:
        return "OpsRAG answer"
    # Strip fenced code blocks entirely from the fallback -- too noisy.
    work = _FENCED_CODE_RE.sub("", answer)
    work = _LINK_RE.sub(lambda m: m.group(1), work)
    work = _BOLD_RE.sub(lambda m: m.group(1), work)
    work = _ITALIC_RE.sub(lambda m: m.group(1), work)
    work = re.sub(r"[#*_`>]", "", work)
    work = re.sub(r"\s+", " ", work).strip()
    if len(work) > 200:
        work = work[:197] + "..."
    return work or "OpsRAG answer"
