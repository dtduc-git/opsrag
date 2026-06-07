"""Runbook parser -- tags sections with operational semantics.

Builds on GenericMarkdownParser but labels sections as procedure_step,
prerequisites, verification, rollback, etc., so downstream nodes
(generator, grader) can weight them.
"""
from __future__ import annotations

import re

from opsrag.interfaces.parser import DocSection, DocType, ParsedDocument
from opsrag.interfaces.scm import RepoFile
from opsrag.parsers.markdown import GenericMarkdownParser

_SECTION_TYPE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(overview|summary|purpose)", re.I), "overview"),
    (re.compile(r"(prereq|before|requirement)", re.I), "prerequisites"),
    (re.compile(r"(procedure|steps?|how\s*to|runbook)", re.I), "procedure_step"),
    (re.compile(r"(verify|verification|validate|check)", re.I), "verification"),
    (re.compile(r"(rollback|revert|recover)", re.I), "rollback"),
    (re.compile(r"(troubleshoot|debug|diagnose)", re.I), "troubleshooting"),
    (re.compile(r"(escalat|on[-\s]?call|contact)", re.I), "escalation"),
]

_RUNBOOK_PATH_HINTS = ("runbook", "sop", "ops/")


def _tag_section(heading: str) -> str:
    for rx, label in _SECTION_TYPE_RULES:
        if rx.search(heading):
            return label
    return "generic"


class RunbookParser:
    def __init__(self) -> None:
        self._base = GenericMarkdownParser()

    def supports(self, file_path: str, content: str) -> bool:
        if not self._base.supports(file_path, content):
            return False
        low = file_path.lower()
        if any(h in low for h in _RUNBOOK_PATH_HINTS):
            return True
        head = content[:512].lower()
        return "runbook" in head

    def detect_doc_type(self, file: RepoFile) -> DocType:
        return DocType.RUNBOOK

    def parse(self, file: RepoFile) -> ParsedDocument:
        doc = self._base.parse(file)
        doc.doc_type = DocType.RUNBOOK
        tagged_sections: list[DocSection] = []
        for s in doc.sections:
            tagged_sections.append(
                DocSection(
                    heading=s.heading,
                    content=s.content,
                    level=s.level,
                    section_type=_tag_section(s.heading),
                    breadcrumb=s.breadcrumb,  # keep H1->H2->H3 scope (was dropped)
                )
            )
        doc.sections = tagged_sections
        doc.metadata["runbook"] = True
        return doc
