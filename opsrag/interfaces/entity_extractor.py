"""Entity extractor interface."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from opsrag.interfaces.graphstore import Entity, Relationship
from opsrag.interfaces.parser import ParsedDocument


@dataclass
class ExtractionResult:
    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)


@runtime_checkable
class EntityExtractor(Protocol):
    async def extract(
        self,
        doc: ParsedDocument,
        existing_entities: list[Entity] | None = None,
    ) -> ExtractionResult: ...
