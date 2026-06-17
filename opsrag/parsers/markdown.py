"""Generic Markdown parser -- lightweight, no external dependencies.

Extracts title, heading-based sections, and markdown link references.
"""
from __future__ import annotations

import re

from opsrag.ingestion.metadata import apply_provenance
from opsrag.interfaces.parser import DocSection, DocType, ParsedDocument
from opsrag.interfaces.scm import RepoFile

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
_MD_EXTENSIONS = (".md", ".markdown", ".mdx")

# A fenced-code marker line: starts (after optional indent) with >=3 backticks
# or >=3 tildes. Used to pre-scan fenced char ranges so an ATX `#`-prefixed
# line INSIDE a ``` block (a shell comment, a Python comment, a Dockerfile
# directive) is not mis-parsed as a markdown heading and split out as its own
# section (M5).
_FENCE_RE = re.compile(r"^[ \t]*(?:`{3,}|~{3,})", re.MULTILINE)


def _fenced_ranges(content: str) -> list[tuple[int, int]]:
    """Char ranges [start, end) covered by fenced code blocks.

    A fence opens on a line beginning with ``` / ~~~ and closes only on a later
    line with the SAME marker char and a run >= the opening length (CommonMark)
    -- a ``` block is NOT closed by ~~~, and a foreign fence line inside the
    block is treated as content, not a close. Ranges span from the opening
    fence's line start through the end of the closing fence's line, so any
    heading-like line strictly inside is shielded. Unterminated -> EOF.
    """
    ranges: list[tuple[int, int]] = []
    n = len(content)
    open_start: int | None = None
    open_char = ""
    open_len = 0
    for m in _FENCE_RE.finditer(content):
        run = content[m.start():m.end()].lstrip(" \t")
        char, length = run[0], len(run)
        if open_start is None:
            open_start, open_char, open_len = m.start(), char, length
        elif char == open_char and length >= open_len:
            line_end = content.find("\n", m.end())
            end = (line_end + 1) if line_end != -1 else n
            ranges.append((open_start, end))
            open_start = None
        # else: a different-marker / shorter fence line inside the open block
        # is part of the code content -- ignore it.
    if open_start is not None:
        ranges.append((open_start, n))
    return ranges


def _outside_fences(pos: int, ranges: list[tuple[int, int]]) -> bool:
    """True if char offset ``pos`` is not inside any fenced range."""
    for start, end in ranges:
        if start <= pos < end:
            return False
    return True


class GenericMarkdownParser:
    """Parses any Markdown file into sections by heading."""

    def supports(self, file_path: str, content: str) -> bool:
        return file_path.lower().endswith(_MD_EXTENSIONS)

    def detect_doc_type(self, file: RepoFile) -> DocType:
        path = file.path.lower()
        if "runbook" in path:
            return DocType.RUNBOOK
        if "postmortem" in path or "post-mortem" in path or "incident" in path:
            return DocType.POSTMORTEM
        if "adr" in path or "/decisions/" in path:
            return DocType.ADR
        if "architecture" in path:
            return DocType.ARCHITECTURE
        return DocType.GENERIC_MARKDOWN

    def parse(self, file: RepoFile) -> ParsedDocument:
        content = file.content
        doc_type = self.detect_doc_type(file)
        sections = self._extract_sections(content)
        title = sections[0].heading if sections else file.path.rsplit("/", 1)[-1]
        references = list(dict.fromkeys(_LINK_RE.findall(content)))

        metadata = {
            "repo": file.repo,
            "branch": file.branch,
            "path": file.path,
            "sha": file.sha,
        }
        # Provenance/identity facets (source_system, updated_at, service,
        # url/author/owner_team where the source carries them). Additive,
        # never clobbers the keys above.
        apply_provenance(metadata, file)
        return ParsedDocument(
            content=content,
            doc_type=doc_type,
            title=title,
            source=file,
            metadata=metadata,
            sections=sections,
            references=references,
        )

    @staticmethod
    def _extract_sections(content: str) -> list[DocSection]:
        # Only accept headings OUTSIDE fenced code blocks: a `#`-prefixed line
        # inside a ``` block (shell/Python comment, Dockerfile directive) is
        # NOT a markdown heading and must not split the fence into sections (M5).
        fenced = _fenced_ranges(content)
        matches = [
            m for m in _HEADING_RE.finditer(content)
            if _outside_fences(m.start(), fenced)
        ]
        if not matches:
            return [DocSection(heading="", content=content.strip(), level=0)]

        sections: list[DocSection] = []
        # Preamble before the FIRST heading (TL;DR / summary intros, top-of-file
        # Terraform comment blocks) was silently dropped -- the loop below starts
        # each section at a heading's end, so content[:first_heading] never got
        # chunked/embedded/indexed. Emit it as a leading headingless section.
        preamble = content[: matches[0].start()].strip()
        if preamble:
            sections.append(DocSection(heading="", content=preamble, level=0))
        # Stack of (level, heading) ancestors so each section carries its full
        # H1 -> H2 -> H3 breadcrumb (was flattened: H2/H3 became sibling parents
        # and the H1 scope never reached the embedding).
        stack: list[tuple[int, str]] = []
        for i, m in enumerate(matches):
            level = len(m.group(1))
            heading = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            body_start = m.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            body = content[body_start:body_end].strip()
            sections.append(DocSection(
                heading=heading, content=body, level=level,
                breadcrumb=[h for _lvl, h in stack],
            ))
        return sections
