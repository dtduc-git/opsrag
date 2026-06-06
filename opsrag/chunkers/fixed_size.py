"""Fixed-size chunker with character-based windows and overlap.

Uses a rough 4-chars-per-token estimate to avoid tokenizer dependencies.
"""
from __future__ import annotations

import hashlib

from opsrag.interfaces.chunker import Chunk
from opsrag.interfaces.parser import ParsedDocument

_CHARS_PER_TOKEN = 4


class FixedSizeChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._char_size = chunk_size * _CHARS_PER_TOKEN
        self._char_overlap = overlap * _CHARS_PER_TOKEN

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        text = doc.content
        if not text.strip():
            return []

        chunks: list[Chunk] = []
        start = 0
        idx = 0
        step = self._char_size - self._char_overlap
        while start < len(text):
            end = min(start + self._char_size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunk_id = self._make_id(doc, idx, piece)
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        content=piece,
                        doc_type=doc.doc_type,
                        source_path=doc.source.path,
                        repo=doc.source.repo,
                        metadata={
                            **doc.metadata,
                            "title": doc.title,
                            "chunk_index": idx,
                        },
                        parent_chunk_id=None,
                        chunk_type="child",
                        token_count=max(1, len(piece) // _CHARS_PER_TOKEN),
                    )
                )
                idx += 1
            if end == len(text):
                break
            start += step
        return chunks

    @staticmethod
    def _make_id(doc: ParsedDocument, idx: int, content: str) -> str:
        h = hashlib.sha1(
            f"{doc.source.repo}:{doc.source.path}:{idx}:{content[:64]}".encode()
        ).hexdigest()[:16]
        return f"{doc.source.repo}:{doc.source.path}:{idx}:{h}"
