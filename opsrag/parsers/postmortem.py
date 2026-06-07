"""Postmortem parser -- extracts root cause / timeline / impact sections.

Tags sections so the graph extractor (Phase 3) and generator can identify
what happened, when, and why.
"""
from __future__ import annotations

import re

from opsrag.interfaces.parser import DocSection, DocType, ParsedDocument
from opsrag.interfaces.scm import RepoFile
from opsrag.parsers.markdown import GenericMarkdownParser

_SECTION_TYPE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(summary|tl[; ]?dr|overview)", re.I), "summary"),
    (re.compile(r"(impact|blast.?radius|customer)", re.I), "impact"),
    (re.compile(r"(timeline|chronology|sequence)", re.I), "timeline"),
    (re.compile(r"(root\s*cause|rca|contributing)", re.I), "root_cause"),
    (re.compile(r"(detection|alert|monitor)", re.I), "detection"),
    (re.compile(r"(mitigation|resolution|fix)", re.I), "mitigation"),
    (re.compile(r"(action\s*item|follow.?up|prevention)", re.I), "action_items"),
    (re.compile(r"(lessons|what\s*went\s*(well|wrong))", re.I), "lessons_learned"),
]

_POSTMORTEM_PATH_HINTS = ("postmortem", "post-mortem", "incidents/", "incident/")


def _tag_section(heading: str) -> str:
    for rx, label in _SECTION_TYPE_RULES:
        if rx.search(heading):
            return label
    return "generic"


class PostmortemParser:
    def __init__(self) -> None:
        self._base = GenericMarkdownParser()

    def supports(self, file_path: str, content: str) -> bool:
        if not self._base.supports(file_path, content):
            return False
        low = file_path.lower()
        if any(h in low for h in _POSTMORTEM_PATH_HINTS):
            return True
        head = content[:512].lower()
        return "root cause" in head or "post-mortem" in head or "postmortem" in head

    def detect_doc_type(self, file: RepoFile) -> DocType:
        return DocType.POSTMORTEM

    def parse(self, file: RepoFile) -> ParsedDocument:
        doc = self._base.parse(file)
        doc.doc_type = DocType.POSTMORTEM
        tagged: list[DocSection] = []
        for s in doc.sections:
            tagged.append(
                DocSection(
                    heading=s.heading,
                    content=s.content,
                    level=s.level,
                    section_type=_tag_section(s.heading),
                    breadcrumb=s.breadcrumb,  # keep H1->H2->H3 scope (was dropped)
                )
            )
        doc.sections = tagged
        doc.metadata["postmortem"] = True
        return doc
