"""Chunking strategy interface."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from opsrag.interfaces.parser import DocType, ParsedDocument


@dataclass
class Chunk:
    id: str
    content: str
    doc_type: DocType
    source_path: str
    repo: str
    metadata: dict = field(default_factory=dict)
    parent_chunk_id: str | None = None
    chunk_type: str = "child"  # "parent" | "child"
    token_count: int = 0


@runtime_checkable
class ChunkingStrategy(Protocol):
    def chunk(self, doc: ParsedDocument) -> list[Chunk]: ...
