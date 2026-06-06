"""Render an incident (+ optional post-mortem) to Markdown.

One incident produces one document. The post-mortem, when present,
is appended as a section so retrieval surfaces both summary and the
deep root-cause writeup as a single coherent unit.

Post-mortem content arrives as HTML (Rootly's editor stores rich text
this way). We do a minimal regex-based HTML->Markdown pass -- Rootly's
post-mortems use a small subset (h1-h4, p, ul/ol/li, strong/em, br,
code, pre, table) so a real HTML parser is overkill.
"""
from __future__ import annotations

import re
from datetime import UTC
from html import unescape

from opsrag.sources.rootly.client import Incident, PostMortem

# Same redaction set as Slack -- incidents can include credential paste-ins.
_REDACTIONS = [
    (re.compile(r"xox[abps]-[A-Za-z0-9-]{10,}"), "[redacted-slack-token]"),
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "[redacted-github-token]"),
    (re.compile(r"gho_[A-Za-z0-9]{20,}"), "[redacted-github-oauth]"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[redacted-aws-key]"),
    (re.compile(r"rootly_[A-Za-z0-9]{32,}"), "[redacted-rootly-token]"),
]


def _redact(text: str) -> str:
    for pat, repl in _REDACTIONS:
        text = pat.sub(repl, text)
    return text


# -- HTML -> Markdown ------------------------------------------------
# Order matters: replace block elements first (preserve newlines) then
# inline. We tolerate broken HTML -- Rootly's editor sometimes emits
# unclosed tags.

_BLOCK_REPLACEMENTS = [
    (re.compile(r"</?(?:html|body|div|section|article)[^>]*>", re.I), ""),
    (re.compile(r"<br\s*/?>", re.I), "\n"),
    (re.compile(r"<hr\s*/?>", re.I), "\n---\n"),
    (re.compile(r"<h1[^>]*>(.*?)</h1>", re.I | re.S), r"\n# \1\n"),
    (re.compile(r"<h2[^>]*>(.*?)</h2>", re.I | re.S), r"\n## \1\n"),
    (re.compile(r"<h3[^>]*>(.*?)</h3>", re.I | re.S), r"\n### \1\n"),
    (re.compile(r"<h4[^>]*>(.*?)</h4>", re.I | re.S), r"\n#### \1\n"),
    (re.compile(r"<h5[^>]*>(.*?)</h5>", re.I | re.S), r"\n##### \1\n"),
    (re.compile(r"<h6[^>]*>(.*?)</h6>", re.I | re.S), r"\n###### \1\n"),
    (re.compile(r"<p[^>]*>(.*?)</p>", re.I | re.S), r"\n\1\n"),
    (re.compile(r"<blockquote[^>]*>(.*?)</blockquote>", re.I | re.S), r"\n> \1\n"),
    (re.compile(r"<pre[^>]*>(.*?)</pre>", re.I | re.S), r"\n```\n\1\n```\n"),
    (re.compile(r"<li[^>]*>(.*?)</li>", re.I | re.S), r"- \1\n"),
    (re.compile(r"</?(?:ul|ol)[^>]*>", re.I), "\n"),
]

_INLINE_REPLACEMENTS = [
    (re.compile(r"<(?:strong|b)[^>]*>(.*?)</(?:strong|b)>", re.I | re.S), r"**\1**"),
    (re.compile(r"<(?:em|i)[^>]*>(.*?)</(?:em|i)>", re.I | re.S), r"*\1*"),
    (re.compile(r"<code[^>]*>(.*?)</code>", re.I | re.S), r"`\1`"),
    (re.compile(r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.I | re.S), r"[\2](\1)"),
    # Strip all remaining tags. We've handled the meaningful ones.
    (re.compile(r"<[^>]+>"), ""),
]

_WHITESPACE_COLLAPSE = re.compile(r"\n{3,}")


def html_to_markdown(html: str) -> str:
    if not html:
        return ""
    out = html
    for pat, repl in _BLOCK_REPLACEMENTS:
        out = pat.sub(repl, out)
    for pat, repl in _INLINE_REPLACEMENTS:
        out = pat.sub(repl, out)
    out = unescape(out)
    out = _WHITESPACE_COLLAPSE.sub("\n\n", out)
    return out.strip()


# -- document rendering --------------------------------------------
def _ts(dt) -> str:
    return dt.astimezone(UTC).isoformat() if dt else ""


def _yaml_list(items) -> str:
    if not items:
        return "[]"
    return "[" + ", ".join(repr(s) for s in items) + "]"


def render_incident(inc: Incident, post_mortem: PostMortem | None = None) -> str:
    """Return a Markdown document covering one incident.

    Frontmatter carries the structured fields (status, severity,
    timestamps, services, teams) -- useful for downstream filtering.
    Body covers narrative content: summary, mitigation, resolution,
    and (when present) the post-mortem.
    """
    frontmatter = (
        "---\n"
        "type: rootly-incident\n"
        f"incident_id: \"{inc.id}\"\n"
        f"sequential_id: {inc.sequential_id if inc.sequential_id is not None else 'null'}\n"
        f"status: \"{inc.status}\"\n"
        f"severity: \"{inc.severity_name}\"\n"
        f"started_at: \"{_ts(inc.started_at)}\"\n"
        f"detected_at: \"{_ts(inc.detected_at)}\"\n"
        f"mitigated_at: \"{_ts(inc.mitigated_at)}\"\n"
        f"resolved_at: \"{_ts(inc.resolved_at)}\"\n"
        f"slack_channel: \"{inc.slack_channel_name}\"\n"
        f"source_url: \"{inc.url}\"\n"
        f"services: {_yaml_list(inc.services)}\n"
        f"teams: {_yaml_list(inc.teams)}\n"
        f"environments: {_yaml_list(inc.environments)}\n"
        f"incident_types: {_yaml_list(inc.incident_types)}\n"
        f"causes: {_yaml_list(inc.causes)}\n"
        f"labels: {_yaml_list(inc.labels)}\n"
        f"has_post_mortem: {'true' if post_mortem else 'false'}\n"
        "---\n\n"
    )

    seq = f"#{inc.sequential_id} " if inc.sequential_id else ""
    body: list[str] = [f"# {seq}{inc.title}\n"]

    # Top-level context line (severity + impacted services) so a chunk
    # that lands on a child node still carries enough context to be
    # cited usefully.
    context_bits: list[str] = []
    if inc.severity_name:
        context_bits.append(f"**Severity:** {inc.severity_name}")
    if inc.services:
        context_bits.append(f"**Services:** {', '.join(inc.services)}")
    if inc.environments:
        context_bits.append(f"**Environments:** {', '.join(inc.environments)}")
    if context_bits:
        body.append(" | ".join(context_bits) + "\n")

    if inc.summary:
        body.append("## Summary\n\n" + _redact(inc.summary).strip() + "\n")

    # Timeline -- render as a small bullet list so chunkers and humans
    # both find it readable. Skip lines with no timestamp.
    timeline_rows = [
        ("Started", inc.started_at),
        ("Detected", inc.detected_at),
        ("Acknowledged", inc.acknowledged_at),
        ("Mitigated", inc.mitigated_at),
        ("Resolved", inc.resolved_at),
        ("Closed", inc.closed_at),
    ]
    timeline_lines = [
        f"- **{label}**: {_ts(when)}"
        for label, when in timeline_rows
        if when is not None
    ]
    if timeline_lines:
        body.append("## Timeline\n\n" + "\n".join(timeline_lines) + "\n")

    if inc.mitigation_message:
        body.append("## Mitigation\n\n" + _redact(inc.mitigation_message).strip() + "\n")
    if inc.resolution_message:
        body.append("## Resolution\n\n" + _redact(inc.resolution_message).strip() + "\n")
    if inc.cancellation_message:
        body.append("## Cancellation note\n\n" + _redact(inc.cancellation_message).strip() + "\n")

    if post_mortem and post_mortem.content_html:
        pm_md = html_to_markdown(post_mortem.content_html)
        if pm_md:
            body.append(
                "## Post-mortem"
                + (f" -- {post_mortem.title}" if post_mortem.title else "")
                + "\n\n"
                + _redact(pm_md)
                + "\n"
            )

    return frontmatter + "\n".join(body)
