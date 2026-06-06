"""ADF -> Markdown rendering.

Atlassian Document Format is a JSON tree of typed nodes. We walk it
recursively and emit Markdown that the existing markdown chunker can
consume as-is.

Coverage policy:
- Block nodes (heading, paragraph, list, table, codeBlock, panel,
  rule, expand) -- explicit handlers.
- Inline marks (strong, em, code, link, underline, strike) -- composed.
- Unknown nodes -- render `[adf:<type>]` placeholder + log once per
  process so we can spot new ADF nodes Atlassian rolls out without
  silently swallowing content.
- Macros (`extension` / `bodiedExtension`) -- render `[macro: <key>]`
  placeholder. Allowlist a small set (`toc`, `decision`, `status`,
  `info`, `note`, `warning`) to render their inner body.
- Media (`mediaSingle`, `mediaInline`) -- render `[attachment: <name>]`.
  Never fetch binary content.

This is intentionally NOT a 1:1 lossless renderer. The goal is to feed
the embedder text that captures the page's information content; a
table that loses cell merges or a `decision` macro that drops its
status pill is fine if the body text survives.

Public API:
- `render_adf(doc) -> str` -- entry point.
- `render_page(page) -> str` -- wraps with YAML frontmatter for the
  Markdown parser's existing schema.
"""
from __future__ import annotations

import logging

import yaml

from opsrag.sources.confluence.client import Page

_log = logging.getLogger("opsrag.confluence.adf")

# We log unknown ADF / extension keys ONCE per process so the warning
# stream doesn't drown in repeats over a 5000-page crawl.
_seen_unknown: set[str] = set()


def _warn_once(category: str, key: str) -> None:
    tag = f"{category}:{key}"
    if tag in _seen_unknown:
        return
    _seen_unknown.add(tag)
    _log.warning("confluence-adf: unhandled %s=%r -- rendered as placeholder", category, key)


# -- Inline rendering --

def _render_text(node: dict) -> str:
    """A single `text` node, applying any `marks` (bold, em, code, link)."""
    text = node.get("text", "")
    if not text:
        return ""
    marks = node.get("marks") or []
    # Order matters for nesting -- apply innermost (code) first, outermost
    # (link) last so the wrapping reads correctly.
    has_code = any(m.get("type") == "code" for m in marks)
    has_strong = any(m.get("type") == "strong" for m in marks)
    has_em = any(m.get("type") == "em" for m in marks)
    has_strike = any(m.get("type") == "strike" for m in marks)
    has_underline = any(m.get("type") == "underline" for m in marks)
    link = next((m for m in marks if m.get("type") == "link"), None)

    out = text
    if has_code:
        # Escape backticks inside inline code by switching to double-tick.
        if "`" in out:
            out = f"`` {out} ``"
        else:
            out = f"`{out}`"
    if has_strike:
        out = f"~~{out}~~"
    if has_underline:
        # Markdown has no native underline; use HTML so the chunker keeps
        # it but readers see something.
        out = f"<u>{out}</u>"
    if has_em:
        out = f"*{out}*"
    if has_strong:
        out = f"**{out}**"
    if link:
        href = (link.get("attrs") or {}).get("href", "")
        if href:
            out = f"[{out}]({href})"
    return out


def _render_inline(nodes: list) -> str:
    """Concat inline-level nodes inside a block (paragraph, heading, ...)."""
    parts: list[str] = []
    for n in nodes or []:
        t = n.get("type")
        if t == "text":
            parts.append(_render_text(n))
        elif t == "hardBreak":
            parts.append("  \n")     # Markdown hard line-break
        elif t == "mention":
            attrs = n.get("attrs") or {}
            display = attrs.get("text") or attrs.get("displayName") or attrs.get("id") or "user"
            # Drop accountId -- it's noise for retrieval.
            parts.append(f"@{display}")
        elif t in ("inlineCard", "blockCard"):
            attrs = n.get("attrs") or {}
            url = attrs.get("url") or ""
            parts.append(f"[link]({url})" if url else "[link]")
        elif t == "emoji":
            attrs = n.get("attrs") or {}
            shortname = attrs.get("shortName") or attrs.get("text") or "emoji"
            parts.append(shortname if shortname.startswith(":") else f":{shortname}:")
        elif t == "date":
            attrs = n.get("attrs") or {}
            ts = attrs.get("timestamp") or ""
            parts.append(f"[date:{ts}]" if ts else "")
        elif t == "status":
            attrs = n.get("attrs") or {}
            label = attrs.get("text") or ""
            parts.append(f"[{label}]" if label else "")
        else:
            _warn_once("inline-node", t or "<missing-type>")
            parts.append(f"[adf:{t}]")
    return "".join(parts)


# -- Block rendering --

def _render_paragraph(node: dict) -> str:
    return _render_inline(node.get("content") or []).rstrip() + "\n\n"


def _render_heading(node: dict) -> str:
    level = max(1, min(6, (node.get("attrs") or {}).get("level", 1)))
    text = _render_inline(node.get("content") or []).strip()
    return f"{'#' * level} {text}\n\n"


def _render_list(node: dict, ordered: bool, depth: int = 0) -> str:
    indent = "  " * depth
    lines: list[str] = []
    for i, item in enumerate(node.get("content") or [], start=1):
        if item.get("type") != "listItem":
            continue
        marker = f"{i}." if ordered else "-"
        # A listItem's children are blocks (paragraph + nested lists).
        item_blocks = item.get("content") or []
        if not item_blocks:
            lines.append(f"{indent}{marker} ")
            continue
        # First block becomes the inline text on the marker line.
        first = item_blocks[0]
        if first.get("type") == "paragraph":
            inline = _render_inline(first.get("content") or []).strip()
            lines.append(f"{indent}{marker} {inline}")
        else:
            rendered = _render_block(first, depth=depth + 1).rstrip()
            lines.append(f"{indent}{marker} {rendered.lstrip()}")
        # Subsequent blocks are indented under the item.
        for blk in item_blocks[1:]:
            if blk.get("type") in ("bulletList", "orderedList"):
                lines.append(_render_list(
                    blk,
                    ordered=blk.get("type") == "orderedList",
                    depth=depth + 1,
                ).rstrip())
            else:
                rendered = _render_block(blk, depth=depth + 1).rstrip()
                if rendered:
                    lines.append(f"{indent}  {rendered}")
    return "\n".join(lines) + ("\n\n" if depth == 0 else "\n")


def _render_code_block(node: dict) -> str:
    attrs = node.get("attrs") or {}
    lang = attrs.get("language") or ""
    # Code block content is plain text nodes -- strip marks, just take text.
    parts = [n.get("text", "") for n in (node.get("content") or []) if n.get("type") == "text"]
    body = "".join(parts).rstrip()
    return f"```{lang}\n{body}\n```\n\n"


def _render_table(node: dict) -> str:
    """Render an ADF table as a Markdown pipe-table.

    Lossy on merged cells / nested blocks inside cells -- we flatten to
    a single line per cell. Acceptable: retrieval cares about the text,
    not visual fidelity.
    """
    rows: list[list[str]] = []
    header_row_idx: int | None = None
    for r_idx, row in enumerate(node.get("content") or []):
        if row.get("type") != "tableRow":
            continue
        cells: list[str] = []
        is_header = False
        for cell in row.get("content") or []:
            ctype = cell.get("type")
            if ctype not in ("tableCell", "tableHeader"):
                continue
            if ctype == "tableHeader":
                is_header = True
            # A cell's content is a list of blocks; flatten to single-line text.
            cell_blocks = cell.get("content") or []
            cell_text_parts: list[str] = []
            for blk in cell_blocks:
                if blk.get("type") == "paragraph":
                    cell_text_parts.append(_render_inline(blk.get("content") or []).strip())
                else:
                    cell_text_parts.append(_render_block(blk, depth=0).strip())
            cell_str = " ".join(p for p in cell_text_parts if p).replace("|", "\\|").replace("\n", " ")
            cells.append(cell_str)
        if cells:
            rows.append(cells)
            if is_header and header_row_idx is None:
                header_row_idx = r_idx if not rows[:-1] else len(rows) - 1
    if not rows:
        return ""
    # Equalize column counts (defensive against malformed tables).
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    # Use first row as header if no explicit header detected.
    if header_row_idx is None:
        header_row_idx = 0
    out: list[str] = []
    for i, row in enumerate(rows):
        out.append("| " + " | ".join(row) + " |")
        if i == header_row_idx:
            out.append("| " + " | ".join(["---"] * width) + " |")
    return "\n".join(out) + "\n\n"


def _render_panel(node: dict) -> str:
    """ADF panel (info/note/warning/success/error) -> Markdown blockquote."""
    attrs = node.get("attrs") or {}
    panel_type = attrs.get("panelType", "info")
    label = {
        "info": "Note",
        "note": "Note",
        "tip": "Tip",
        "warning": "Warning",
        "error": "Error",
        "success": "Success",
    }.get(panel_type, "Note")
    # Render inner blocks then prefix every line with "> ".
    inner = "".join(_render_block(blk, depth=0) for blk in (node.get("content") or [])).rstrip()
    if not inner:
        return ""
    quoted = "\n".join(f"> {line}" if line else ">" for line in inner.split("\n"))
    return f"> **{label}:**\n{quoted}\n\n"


def _render_expand(node: dict) -> str:
    """`expand` / `nestedExpand` -- drop the collapse, keep the body.

    Title (if present) becomes a small heading so it still anchors the body.
    """
    attrs = node.get("attrs") or {}
    title = (attrs.get("title") or "").strip()
    body = "".join(_render_block(blk, depth=0) for blk in (node.get("content") or []))
    if title:
        return f"**{title}**\n\n{body}"
    return body


# Allowlist of macros where the inner body is worth keeping.
_BODIED_MACRO_PASSTHROUGH = {"info", "note", "warning", "tip", "success", "error", "expand"}


def _render_extension(node: dict) -> str:
    """`extension` / `bodiedExtension` / `inlineExtension` -- Confluence macros.

    Atlassian macros come through as opaque extension nodes. We can't
    re-execute them, but we can:
      - Render the inner body for known passthrough macros.
      - Render a placeholder for everything else, logged once.
    """
    attrs = node.get("attrs") or {}
    key = (attrs.get("extensionKey") or attrs.get("extensionType") or "").lower()
    body_blocks = node.get("content") or []
    if key in _BODIED_MACRO_PASSTHROUGH and body_blocks:
        return "".join(_render_block(blk, depth=0) for blk in body_blocks)
    if body_blocks and key in {"toc", "decision", "decision-list"}:
        # Pass through -- the body is the value.
        return "".join(_render_block(blk, depth=0) for blk in body_blocks)
    _warn_once("macro", key or "<missing-key>")
    return f"[macro: {key or 'unknown'}]\n\n"


def _render_media(node: dict) -> str:
    """mediaSingle / mediaInline / mediaGroup -- reference-only.

    Never fetch binary; just include the filename so the page text
    isn't completely empty where an image used to be.
    """
    name = ""
    for child in (node.get("content") or []):
        attrs = child.get("attrs") or {}
        name = attrs.get("alt") or attrs.get("collection") or attrs.get("id") or name
    return f"[attachment: {name or 'unnamed'}]\n\n"


# -- Block dispatch --

def _render_block(node: dict, depth: int = 0) -> str:
    t = node.get("type")
    if t == "doc":
        # Top-level wrapper -- walk children.
        return "".join(_render_block(c, depth) for c in (node.get("content") or []))
    if t == "paragraph":
        return _render_paragraph(node)
    if t == "heading":
        return _render_heading(node)
    if t == "bulletList":
        return _render_list(node, ordered=False, depth=depth)
    if t == "orderedList":
        return _render_list(node, ordered=True, depth=depth)
    if t == "codeBlock":
        return _render_code_block(node)
    if t == "blockquote":
        inner = "".join(
            _render_block(c, depth=0) for c in (node.get("content") or [])
        ).rstrip()
        return "\n".join(f"> {line}" for line in inner.split("\n")) + "\n\n"
    if t == "table":
        return _render_table(node)
    if t == "panel":
        return _render_panel(node)
    if t in ("expand", "nestedExpand"):
        return _render_expand(node)
    if t in ("extension", "bodiedExtension", "inlineExtension"):
        return _render_extension(node)
    if t in ("mediaSingle", "mediaInline", "mediaGroup", "media"):
        return _render_media(node)
    if t == "rule":
        return "---\n\n"
    if t == "listItem":
        # Should normally be reached via _render_list, but render as
        # paragraph if encountered orphan.
        return _render_paragraph(node)
    if t == "text":
        # An orphan text node at block level -- wrap in a paragraph.
        return _render_text(node) + "\n\n"
    _warn_once("block-node", t or "<missing-type>")
    return f"[adf:{t}]\n\n"


# -- Public API --

def render_adf(adf_doc: dict | None) -> str:
    """Render an ADF document tree to Markdown.

    Empty / malformed input -> empty string. Don't raise -- ingestion
    treats unparseable bodies as zero-content pages, not failures.
    """
    if not adf_doc:
        return ""
    if not isinstance(adf_doc, dict):
        return ""
    if adf_doc.get("type") != "doc":
        # Some pages have malformed top -- still try to walk if there are
        # children.
        if "content" in adf_doc:
            return "".join(_render_block(c, depth=0) for c in adf_doc.get("content") or [])
        return ""
    return _render_block(adf_doc).rstrip() + "\n"


def render_page(page: Page, *, last_reviewed: str | None = None) -> str:
    """Render a fetched Page to a Markdown document with YAML frontmatter.

    The frontmatter shape is designed to interop with the existing
    markdown parser (`opsrag.parsers.markdown`) so chunks pick up the
    page's identity in their metadata without parser changes.
    """
    body = render_adf(page.body_adf)
    frontmatter = {
        "id": f"confluence-{page.id}",
        "type": "confluence-page",
        "space": page.space_key or "",
        "page_id": str(page.id),
        "page_title": page.title,
        "page_url": page.url,
        "ancestors": list(page.ancestors or []),
        "labels": list(page.labels or []),
        "last_modified": page.last_modified.isoformat() if page.last_modified else "",
        "version": page.version,
    }
    if last_reviewed:
        frontmatter["last_reviewed"] = last_reviewed
    fm_yaml = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
    title_line = f"# {page.title}\n\n" if page.title else ""
    return f"---\n{fm_yaml}\n---\n\n{title_line}{body}"
