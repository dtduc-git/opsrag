"""Hybrid extractor -- metadata rules + structured files + LLM prose.

Three lanes, combined and de-duplicated by deterministic entity ID:

1. **Metadata rule lane** (zero LLM, always on): turns the enriched chunk
   metadata track (`service` / `owner_team` / `environment` / `repo` /
   `services[]`) into Tier-0 SRE entities + relationships. This is the
   load-bearing, high-precision lane -- the metadata is deterministic and
   not free prose, so it is the trustworthy backbone of the graph.

2. **Structured-file lane** (zero LLM): the existing ``RuleBasedExtractor``
   for Terraform / K8s / Helm / Dockerfile content.

3. **Prose LLM lane** (optional, only when ``method`` is ``hybrid``/``llm``):
   ``LLMEntityExtractor`` over prose docs, routed through the cheap
   ``extract`` purpose. Best-effort -- failure returns nothing, never raises.

Security: lane 1 values come from the metadata track (still ultimately
derived from repo paths/frontmatter, so treated as low-trust) and lane 3
values come from chunk prose (attacker-influenceable). BOTH are passed
through ``schema.sanitize_value`` and constrained to the allow-listed
labels/relations. Edges are low-trust soft-boost at retrieval time.
"""
from __future__ import annotations

from opsrag.extractors.llm_extractor import LLMEntityExtractor
from opsrag.extractors.rule_based import RuleBasedExtractor
from opsrag.extractors.schema import (
    make_entity_id,
    sanitize_value,
)
from opsrag.interfaces.entity_extractor import ExtractionResult
from opsrag.interfaces.graphstore import Entity, Relationship
from opsrag.interfaces.llm import LLMProvider
from opsrag.interfaces.parser import DocType, ParsedDocument

_STRUCTURED_TYPES = {
    DocType.TERRAFORM,
    DocType.KUBERNETES,
    DocType.HELM,
    DocType.DOCKERFILE,
}


def entities_from_metadata(
    metadata: dict,
    source_chunk_id: str | None = None,
) -> ExtractionResult:
    """Tier-0 deterministic extraction from the enriched metadata track.

    Reads the facets the metadata track populates -- ``service`` (scalar),
    ``services[]`` (multi-service), ``owner_team``, ``environment``,
    ``repo`` -- and emits SRE entities + the relationships between them:

      - ``Service -[OWNED_BY]-> Team``
      - ``Service -[RUNS_IN]-> Environment``
      - ``Service -[DEFINED_IN]-> Repository``

    Every value is sanitized; empty/oversized values are dropped. IDs are
    the shared deterministic ``label:name:hash`` so the same service merges
    across files (this is exactly why reference-counted delete is required).
    """
    entities: dict[str, Entity] = {}
    rels: list[Relationship] = []

    def _add(label: str, raw_name: object, props: dict | None = None) -> str | None:
        name = sanitize_value(raw_name)
        if not name:
            return None
        eid = make_entity_id(label, name)
        if eid not in entities:
            entities[eid] = Entity(
                id=eid,
                label=label,
                name=name,
                properties=dict(props or {}),
                source_chunk_id=source_chunk_id,
            )
        return eid

    # Collect the service set: scalar `service` + any `services[]`.
    service_names: list[object] = []
    if metadata.get("service"):
        service_names.append(metadata["service"])
    for s in metadata.get("services") or []:
        service_names.append(s)

    team_id = _add("Team", metadata.get("owner_team")) if metadata.get("owner_team") else None
    env_id = _add("Environment", metadata.get("environment")) if metadata.get("environment") else None
    repo_id = _add("Repository", metadata.get("repo")) if metadata.get("repo") else None

    seen_service_ids: set[str] = set()
    for raw_svc in service_names:
        svc_id = _add("Service", raw_svc)
        if not svc_id or svc_id in seen_service_ids:
            continue
        seen_service_ids.add(svc_id)
        # Stamp the source onto each edge's properties so the graph store can
        # reference-count edges (delete only the source that contributed them).
        rprops = {"source_chunk_id": source_chunk_id} if source_chunk_id else {}
        if team_id:
            rels.append(Relationship(svc_id, team_id, "OWNED_BY", dict(rprops)))
        if env_id:
            rels.append(Relationship(svc_id, env_id, "RUNS_IN", dict(rprops)))
        if repo_id:
            rels.append(Relationship(svc_id, repo_id, "DEFINED_IN", dict(rprops)))

    return ExtractionResult(entities=list(entities.values()), relationships=rels)


def _merge(*results: ExtractionResult) -> ExtractionResult:
    """Combine extraction results, de-duplicating entities by ID and
    relationships by (source, target, type)."""
    by_id: dict[str, Entity] = {}
    rels: dict[tuple[str, str, str], Relationship] = {}
    for r in results:
        for e in r.entities:
            if e.id not in by_id:
                by_id[e.id] = e
            elif e.properties:
                # Later lanes can enrich properties of an already-seen entity.
                by_id[e.id].properties.update(e.properties)
        for rel in r.relationships:
            rels[(rel.source_id, rel.target_id, rel.rel_type)] = rel
    return ExtractionResult(entities=list(by_id.values()), relationships=list(rels.values()))


class HybridExtractor:
    """Metadata-rule + structured-file + (optional) LLM-prose extractor.

    ``method`` controls the LLM prose lane:
      - ``"rule_based"`` -> lanes 1+2 only (no LLM call ever)
      - ``"hybrid"`` (default) -> lanes 1+2, plus LLM prose for non-structured
        docs
      - ``"llm"`` -> lanes 1+3 (LLM prose for everything; structured files
        still also get the rule lane for precision)
    """

    def __init__(self, llm: LLMProvider | None = None, method: str = "hybrid"):
        self._rule = RuleBasedExtractor()
        self._llm_ext = LLMEntityExtractor(llm) if llm is not None else None
        self._method = method

    async def extract(
        self,
        doc: ParsedDocument,
        existing_entities: list[Entity] | None = None,
    ) -> ExtractionResult:
        source_id = f"{doc.source.repo}:{doc.source.path}"
        results: list[ExtractionResult] = []

        # Lane 1 -- metadata rules. ParsedDocument.metadata carries the
        # enriched facets (service/owner_team/environment/repo); also seed
        # `repo` from the source if the parser didn't.
        meta = dict(doc.metadata or {})
        meta.setdefault("repo", getattr(doc.source, "repo", None))
        results.append(entities_from_metadata(meta, source_chunk_id=source_id))

        # Lane 2 -- structured files (always run for structured doc types).
        is_structured = doc.doc_type in _STRUCTURED_TYPES
        if is_structured:
            results.append(await self._rule.extract(doc, existing_entities))

        # Lane 3 -- LLM prose (best-effort). Skipped entirely when method is
        # rule_based, when no LLM is wired, or (in hybrid) for structured docs.
        wants_llm = self._method in ("hybrid", "llm")
        if wants_llm and self._llm_ext is not None:
            run_llm = (self._method == "llm") or (not is_structured)
            if run_llm:
                try:
                    results.append(await self._llm_ext.extract(doc, existing_entities))
                except Exception:
                    # Extraction failure must never propagate -- the graph
                    # lane is non-fatal w.r.t. ingestion.
                    pass

        return _merge(*results)

    async def extract_from_metadata(
        self,
        metadata: dict,
        source_chunk_id: str | None = None,
    ) -> ExtractionResult:
        """Direct metadata-only extraction (lane 1).

        Convenience for the ingestion pipeline, which has enriched chunk
        metadata in hand and does not need to re-derive a ParsedDocument.
        """
        return entities_from_metadata(metadata, source_chunk_id=source_chunk_id)
