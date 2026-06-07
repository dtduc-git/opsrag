"""Document parser interface."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from opsrag.interfaces.scm import RepoFile


class DocType(str, Enum):
    RUNBOOK = "runbook"
    POSTMORTEM = "postmortem"
    TERRAFORM = "terraform"
    HELM = "helm"
    KUBERNETES = "kubernetes"
    DOCKERFILE = "dockerfile"
    ALERT_DEFINITION = "alert_definition"
    ARCHITECTURE = "architecture"
    ADR = "adr"
    GENERIC_MARKDOWN = "generic_markdown"
    YAML_CONFIG = "yaml_config"
    # Source code -- distinct types so contextual chunking can render
    # accurate labels ("Python source" vs "Generic config") and so
    # future per-language enrichment (tree-sitter AST parsing for
    # function/class boundaries, see backlog Step 2) can dispatch
    # cleanly off doc_type. Before these existed, all code files were
    # silently labeled YAML_CONFIG by the fallback parser, which made
    # contextual prefixes lie ("YAML config in saas/acme-notes-be/auth.py").
    PYTHON = "python"
    JAVASCRIPT = "javascript"   # covers .js, .jsx, .mjs, .cjs
    TYPESCRIPT = "typescript"   # covers .ts, .tsx
    GO = "go"
    JAVA = "java"               # covers .java, .kt, .kts
    SHELL = "shell"             # covers .sh, .bash


@dataclass
class DocSection:
    heading: str
    content: str
    level: int
    section_type: str = "generic"
    # Full heading ancestry (H1 -> H2 -> H3) for this section. Parsers that
    # understand nesting populate it; others leave it empty and the chunker
    # falls back to [doc.title, heading]. Used to build the chunk heading_path
    # breadcrumb so an H3 chunk still carries its H1/H2 scope into retrieval.
    breadcrumb: list[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    content: str
    doc_type: DocType
    title: str
    source: RepoFile
    metadata: dict = field(default_factory=dict)
    sections: list[DocSection] = field(default_factory=list)
    references: list[str] = field(default_factory=list)


@runtime_checkable
class DocumentParser(Protocol):
    def supports(self, file_path: str, content: str) -> bool: ...
    def parse(self, file: RepoFile) -> ParsedDocument: ...
    def detect_doc_type(self, file: RepoFile) -> DocType: ...
