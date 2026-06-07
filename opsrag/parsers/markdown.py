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
        matches = list(_HEADING_RE.finditer(content))
        if not matches:
            return [DocSection(heading="", content=content.strip(), level=0)]

        sections: list[DocSection] = []
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
