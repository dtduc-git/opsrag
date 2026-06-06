"""LLM-based entity extractor.

Sends document content to an LLM with a constrained schema to pull out
SRE-domain entities (Service, Incident, Runbook, ...) and their relationships.
"""
from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, Field

from opsrag.extractors.schema import (
    make_entity_id,
    normalize_label,
    normalize_rel_type,
    sanitize_properties,
    sanitize_value,
)
from opsrag.interfaces.entity_extractor import ExtractionResult
from opsrag.interfaces.graphstore import Entity, Relationship
from opsrag.interfaces.llm import LLMProvider
from opsrag.interfaces.parser import ParsedDocument

_log = logging.getLogger("opsrag.extractors.llm")

# Hard cap on a single extraction call. Without it a hung provider stalls the
# whole ingest indefinitely (the call had no timeout, no retry). On timeout we
# return empty -- the metadata + structured lanes still populate the graph.
_EXTRACT_TIMEOUT_S = 30.0

_SYSTEM_PROMPT = """Extract operational entities and relationships from the text.

Entity types: Service, Team, Runbook, Incident, Alert, Config, Database,
Infra, Repository, Environment, Person.

Relationship types: DEPENDS_ON, USES_DATABASE, DEPLOYED_ON, OWNED_BY,
HAS_RUNBOOK, HAS_ALERT, CONFIGURED_BY, LIVES_IN, DEPLOYED_TO,
AFFECTED, ROOT_CAUSE, RESOLVED_BY, INVESTIGATED_BY, TRIGGERS,
REFERENCES, MEMBER_OF, ONCALL_FOR, APPLIES_TO, HOSTED_ON.

Return only entities and relationships you can clearly identify from the
text. Prefer precision over recall -- do not guess."""


class _ExtractedEntity(BaseModel):
    name: str
    label: str
    properties: dict = Field(default_factory=dict)


class _ExtractedRelationship(BaseModel):
    source_name: str
    target_name: str
    rel_type: str
    properties: dict = Field(default_factory=dict)


class _ExtractionSchema(BaseModel):
    entities: list[_ExtractedEntity] = Field(default_factory=list)
    relationships: list[_ExtractedRelationship] = Field(default_factory=list)


class LLMEntityExtractor:
    def __init__(self, llm: LLMProvider):
        self._llm = llm

    async def extract(
        self,
        doc: ParsedDocument,
        existing_entities: list[Entity] | None = None,
    ) -> ExtractionResult:
        content = doc.content[:6000]
        existing_hint = ""
        if existing_entities:
            names = ", ".join(e.name for e in existing_entities[:50])
            existing_hint = f"\n\nAlready-known entities (link to these when possible): {names}"

        source_label = f"{doc.source.repo}:{doc.source.path}"
        try:
            result = await asyncio.wait_for(
                self._llm.generate_structured(
                    purpose="extract",  # cheap lane -- see factory.py wiring
                    messages=[
                        {"role": "user", "content": f"Document:\n{content}{existing_hint}"}
                    ],
                    schema=_ExtractionSchema,
                    system_prompt=_SYSTEM_PROMPT,
                ),
                timeout=_EXTRACT_TIMEOUT_S,
            )
        except TimeoutError:
            # Distinguishable from "no entities found" -- a real stall, logged.
            _log.warning(
                "entity extraction timed out after %.0fs for %s -- skipping",
                _EXTRACT_TIMEOUT_S, source_label,
            )
            return ExtractionResult()
        except Exception as exc:
            _log.warning("entity extraction failed for %s: %s", source_label, exc)
            return ExtractionResult()

        source_id = f"{doc.source.repo}:{doc.source.path}"
        # The LLM output is derived from attacker-influenceable chunk text.
        # Constrain labels/relation types to the allow-list (fail-closed --
        # unknown labels are dropped) and sanitize the free-text VALUES so a
        # prompt-injection payload can't smuggle markup/control chars into the
        # graph. Edges are low-trust by construction (soft-boost at retrieval).
        name_to_id: dict[str, str] = {}
        entities: list[Entity] = []
        for raw in result.entities:
            label = normalize_label(raw.label)
            if label is None:
                continue
            name = sanitize_value(raw.name)
            if not name:
                continue
            eid = make_entity_id(label, name)
            # Map the ORIGINAL (unsanitized) name so relationship resolution
            # below can match the LLM's own references.
            name_to_id[raw.name] = eid
            entities.append(
                Entity(
                    id=eid,
                    label=label,
                    name=name,
                    properties=sanitize_properties(raw.properties),
                    source_chunk_id=source_id,
                )
            )

        rels: list[Relationship] = []
        for raw_r in result.relationships:
            rel_type = normalize_rel_type(raw_r.rel_type)
            if rel_type is None:
                continue
            src = name_to_id.get(raw_r.source_name)
            tgt = name_to_id.get(raw_r.target_name)
            if src and tgt:
                props = sanitize_properties(raw_r.properties)
                # Stamp the source for reference-counted edge deletion.
                props["source_chunk_id"] = source_id
                rels.append(
                    Relationship(
                        source_id=src,
                        target_id=tgt,
                        rel_type=rel_type,
                        properties=props,
                    )
                )

        return ExtractionResult(entities=entities, relationships=rels)
