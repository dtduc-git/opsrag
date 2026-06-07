"""Fixed-size chunker with character-based windows and overlap.

Window char budget is the per-content-type chars/token ratio
(opsrag.tokenization.chars_per_token_for) times the token target, so a window
is sized in tokens of THAT content type rather than "average" text.
"""
from __future__ import annotations

import hashlib

from opsrag.interfaces.chunker import Chunk
from opsrag.interfaces.parser import ParsedDocument

# Char budget is per-doc-type (config ~2.5, code ~3.5, prose ~4.0) so a window
# is sized in tokens of THAT content type, not "average" text. See
# opsrag.tokenization (+ the re-index caveat).
from opsrag.tokenization import chars_per_token_for, estimate_tokens


class FixedSizeChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        text = doc.content
        if not text.strip():
            return []

        cpt = chars_per_token_for(doc.doc_type)
        char_size = int(self.chunk_size * cpt)
        char_overlap = int(self.overlap * cpt)

        chunks: list[Chunk] = []
        start = 0
        idx = 0
        step = char_size - char_overlap
        while start < len(text):
            end = min(start + char_size, len(text))
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
                        token_count=estimate_tokens(piece, doc.doc_type),
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
