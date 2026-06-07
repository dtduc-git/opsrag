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
    # Text used for the DENSE embedding ONLY, when it must differ from `content`
    # -- e.g. a contextual `[Context: ...]` prefix that aids semantic matching
    # but must NOT pollute the BM25/FTS lexical index (diluting IDF, double-
    # counting path slugs) or the stored/displayed content. None -> embed
    # `content`. BM25/FTS/payload/display always use `content`.
    embed_content: str | None = None


@runtime_checkable
class ChunkingStrategy(Protocol):
    def chunk(self, doc: ParsedDocument) -> list[Chunk]: ...
