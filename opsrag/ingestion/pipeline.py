"""Ingestion pipeline: SCM -> parse -> chunk -> embed -> vector store.

Within-repo file parallelism via OPSRAG_FILE_PARALLEL (default 4) -- combined
with repo-level concurrency in server.py for end-to-end throughput.

Step 7: an optional `IndexedFilesTracker` lets us short-circuit per-file
work when content_hash matches a previously-indexed version. Disabled when
the tracker is the no-op variant, so behaviour is unchanged for setups
without Postgres.

Graph lane (DESIGN 3 PART 2, re-activated): when BOTH `graph_store` and
`entity_extractor` are wired (provider != none + extraction method != none),
each indexed file's enriched chunk metadata + prose is turned into SRE
entities/relationships and upserted to the graph store. The lane is
strictly non-fatal -- any graph failure is logged and swallowed so it can
never block vector indexing. The original 2026-05-23 removal was operational
(Neo4j Community lacked APOC -> silent-empty graph); that failure mode is now
guarded by a fail-fast APOC check at factory build time, so a misconfigured
graph surfaces loudly at startup instead of silently here.

Delete semantics: graph `delete_by_source` is REFERENCE-COUNTED in the store
-- a single-file reindex removes only the sources that file contributed and
keeps shared cross-file entities/edges that other live sources still
reference. The per-file graph source id is `"{repo}:{path}"` (matching the
extractors' `source_chunk_id`).
"""
from __future__ import annotations

import asyncio
import gc
import hashlib
import logging
import os
from typing import Any

from opsrag.indexed_files.noop import NoopIndexedFilesTracker
from opsrag.ingestion import contextual
from opsrag.ingestion.enrich import enrich_metadata
from opsrag.interfaces.chunker import ChunkingStrategy
from opsrag.interfaces.embedder import EmbeddingProvider
from opsrag.interfaces.entity_extractor import EntityExtractor
from opsrag.interfaces.graphstore import Entity, KnowledgeGraphStore, Relationship
from opsrag.interfaces.indexed_files import IndexedFilesTracker
from opsrag.interfaces.llm import LLMProvider
from opsrag.interfaces.parser import DocType, DocumentParser


def _graph_source_id(repo: str, path: str) -> str:
    """Per-file graph source key -- matches the extractors' source_chunk_id
    (`"{repo}:{path}"`) so reference-counted delete can target exactly the
    entities/edges this file contributed."""
    return f"{repo}:{path}"

# P3 -- DocTypes considered "code" for the dual-write to the code
# collection. Chunks of these types get embedded with the code embedder
# and added to `opsrag_code` (in addition to the main collection).
# Anything else (markdown, helm, terraform, runbooks, ...) stays in the
# main collection only -- generic embedders handle prose well.
_CODE_DOC_TYPES: frozenset[DocType] = frozenset({
    DocType.PYTHON,
    DocType.JAVASCRIPT,
    DocType.TYPESCRIPT,
    DocType.GO,
    DocType.JAVA,
    DocType.SHELL,
})
from opsrag.indexing_tracker import indexing_tracker
from opsrag.interfaces.scm import RepoFile, SCMProvider, WebhookEvent
from opsrag.interfaces.source import SourceProtocol
from opsrag.interfaces.vectorstore import VectorStore

_log = logging.getLogger("opsrag.ingestion")
_DEFAULT_FILE_PARALLEL = 4

# Path-substring blacklist applied AFTER scm.list_files. Catches noise that
# slips past glob whitelists: lockfiles, generated code, build artifacts,
# vendored deps, Django auto-migrations, etc. Substring match is intentional
# -- covers nested paths like `apps/foo/node_modules/bar/index.js` that a
# top-level fnmatch would miss.
_DEFAULT_EXCLUDES: tuple[str, ...] = (
    "/node_modules/", "/__pycache__/", "/.venv/", "/venv/",
    "/dist/", "/build/", "/.next/", "/.nuxt/", "/.turbo/",
    "/vendor/", "/target/", "/.cache/",
    "/migrations/0",          # Django auto-generated migration files
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    ".min.js", ".min.css",
    ".pb.go", "_pb2.py",      # generated protobuf
    # Vendored snapshots inside internal-image-repo. The opsrag image
    # source tree contains a copy of sre-knowledge-base + runbooks at
    # `images/opsrag/{backend,backend,indexer,indexer}/...` so that
    # the container ships with markdown bundled. Those copies are NOT
    # source -- the canonical source is the upstream `sre-knowledge-base`
    # repo, which is indexed separately. Without this exclude we end up
    # with every chunk indexed twice, halving recall in practice.
    "/images/opsrag/",
    # IDE / AI-assistant local config trees. These get committed to source
    # repos (skill caches, editor settings, agent-mode configs) but they
    # are NOT documentation about the service -- they're tool config for a
    # developer's local environment. Indexing them adds noise: queries
    # about a service's *behavior* end up retrieving unrelated skill docs
    # like `marketplace-hosting.md` because the prose looks topical. Seen
    # firsthand 2026-05-18 -- Q1 of a 5-query test regressed because
    # `.claude/skills/...marketplace-hosting.md` chunks beat the actual
    # service config in top-K.
    "/.claude/",              # Claude Code skill caches + agent prompts
    "/.cursor/",              # Cursor IDE config
    "/.vscode/",              # VS Code workspace settings
    "/.idea/",                # JetBrains config
    "/.codeium/",             # Codeium config
)


def _derive_display_name(source_type: str, metadata: dict) -> str | None:
    """Build a human-readable label for the indexing dashboard.

    For sources whose `repo` field is an opaque ID (Slack channel IDs,
    most notably), pull a friendly name out of the document metadata
    so the UI shows "slack:#devops" instead of "slack:CC448TKTQ".
    Returns None when the raw `repo` is already readable (git, Confluence).
    """
    if source_type == "slack":
        ch = metadata.get("channel_name")
        if ch:
            return f"slack:#{ch}"
    if source_type == "rootly":
        # Tracker key is `rootly:<scope>` (e.g. `rootly:default`).
        # Already readable, but adding "incidents" makes it clearer
        # in the indexing nav what kind of content lives there.
        return "rootly:incidents"
    return None


def _is_excluded(path: str) -> bool:
    # Normalise so a top-level "vendor/lib.go" matches the same "/vendor/"
    # rule as a nested "apps/foo/vendor/lib.go".
    normalized = "/" + path
    return any(s in normalized for s in _DEFAULT_EXCLUDES)


class IngestionPipeline:
    def __init__(
        self,
        scm: SCMProvider,
        parsers: list[DocumentParser],
        chunker: ChunkingStrategy,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        graph_store: KnowledgeGraphStore | None = None,
        entity_extractor: EntityExtractor | None = None,
        llm: LLMProvider | None = None,
        indexed_files: IndexedFilesTracker | None = None,
        sources: dict[str, SourceProtocol] | None = None,
        code_embedder: EmbeddingProvider | None = None,
        code_vector_store: VectorStore | None = None,
        light_graph: Any | None = None,
    ):
        self.scm = scm
        # Lightweight entity-graph (Postgres edges) for the entity-expansion
        # retrieval lane. When set, each chunk gets `entity_ids` on its payload
        # + structured edges are upserted here. Independent of the Neo4j graph
        # lane (works with knowledge_graph.provider=none). None -> no-op.
        self.light_graph = light_graph
        self.parsers = parsers
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        # Graph lane (re-activated DESIGN 3 PART 2). Active only when BOTH are
        # wired -- factory passes a NullGraphStore + None extractor for the
        # default (provider=none) deployment, so `_graph_enabled` is False and
        # the lane is a no-op. The lane is additionally non-fatal at every
        # call site so it can never block vector indexing.
        self.graph_store = graph_store
        self.entity_extractor = entity_extractor
        # Optional LLM -- only used when contextual chunking is enabled.
        self.llm = llm
        # Tracker for content-hash-based dedup. Default no-op so existing
        # callers don't need updates and behaviour stays identical.
        self.indexed_files: IndexedFilesTracker = (
            indexed_files or NoopIndexedFilesTracker()
        )
        # Phase 2 -- non-git sources (Confluence, Rootly, Slack, ...) keyed
        # by `source_type`. Empty dict by default so existing git-only
        # deployments are unchanged.
        self.sources: dict[str, SourceProtocol] = sources or {}
        # P3 (2026-05-18) -- optional code-specific embedder + collection.
        # When both are set, code DocType chunks are ALSO embedded with
        # `code_embedder` and upserted into `code_vector_store` (Path Y
        # dual-write design from docs/p3-code-embedder-design.md).
        # When either is None, behavior is identical to pre-P3.
        self.code_embedder = code_embedder
        self.code_vector_store = code_vector_store
        # Lazily-built structured-file extractor for the light-graph lane.
        # Independent of the Neo4j `entity_extractor` (which is None when
        # graph provider=none) -- the light lane needs its own zero-LLM
        # extractor so rich Service/Config/Repository edges are captured even
        # with the graph disabled. Built on first use; reused across files.
        self._light_rule = None

    @property
    def _graph_enabled(self) -> bool:
        """True only when a real graph lane is wired.

        NullGraphStore (the default) is detected by class name so we don't
        even attempt extraction/upsert for the zero-graph deployment.
        """
        if self.graph_store is None or self.entity_extractor is None:
            return False
        return type(self.graph_store).__name__ != "NullGraphStore"

    async def _attach_entity_ids(self, chunks: list, doc, repo: str) -> list:
        """Light-graph lane: derive deterministic entity ids + structured edges
        for the entity-expansion retrieval lane, stamp the ids onto each chunk's
        ``metadata['entity_ids']`` (-> Qdrant payload), and return the de-duped
        edges for the Postgres adjacency table. Two zero-LLM lanes, both
        independent of the Neo4j graph lane:

          1. **Per-chunk metadata rules** -- service/owner_team/environment/repo
             facets the enricher stamped onto each chunk (lane 1, high-precision).
          2. **Structured-file content** -- ``RuleBasedExtractor`` over the parsed
             doc (Terraform/K8s/Helm/Dockerfile + ingress/mesh routing): file-level
             Service/Config/Repository/Cluster entities + their edges
             (LIVES_IN/IN_CLUSTER/routing). These file-level ids are stamped onto
             EVERY chunk of the file, so retrieving any chunk seeds the file's
             entities and the 1-hop expand can reach cross-file neighbors that
             share an entity (e.g. two services using the same image).

        Cheap, no LLM, no Neo4j. Both lanes are non-fatal; a failure in one
        still lets the other contribute. No-op + no cost when the light graph
        isn't wired."""
        if self.light_graph is None:
            return []
        from opsrag.extractors.hybrid import entities_from_metadata

        edges: dict = {}

        # Lane 2 -- structured-file content (once per doc). Built lazily; the
        # extractor is zero-LLM (regex + yaml on the thread pool). Failure is
        # swallowed so lane 1 still runs.
        doc_entity_ids: set[str] = set()
        try:
            if self._light_rule is None:
                from opsrag.extractors.rule_based import RuleBasedExtractor
                self._light_rule = RuleBasedExtractor()
            struct = await self._light_rule.extract(doc)
            doc_entity_ids = {e.id for e in (getattr(struct, "entities", []) or [])}
            for r in getattr(struct, "relationships", []) or []:
                edges[(r.source_id, r.target_id, r.rel_type)] = r
        except Exception as exc:
            _log.warning(
                "light-graph structured lane failed repo=%s: %s -- proceeding",
                repo, exc,
            )

        # Lane 1 -- per-chunk metadata rules. Merge the file-level structured
        # ids onto every chunk.
        for c in chunks:
            try:
                meta = dict(getattr(c, "metadata", None) or {})
                meta.setdefault("repo", repo)
                res = entities_from_metadata(meta, source_chunk_id=getattr(c, "id", None))
                ids = {e.id for e in (getattr(res, "entities", []) or [])}
                ids |= doc_entity_ids
                if ids:
                    c.metadata["entity_ids"] = sorted(ids)
                for r in getattr(res, "relationships", []) or []:
                    edges[(r.source_id, r.target_id, r.rel_type)] = r
            except Exception:
                # Even if the metadata lane fails for this chunk, still attach
                # the file-level structured ids so the chunk is reachable.
                if doc_entity_ids:
                    try:
                        c.metadata["entity_ids"] = sorted(doc_entity_ids)
                    except Exception:
                        pass
                continue
        return list(edges.values())

    async def _graph_delete_by_source(self, repo: str, path: str) -> None:
        """Reference-counted graph delete sweep for one file. Non-fatal.

        Removes only the entities/edges THIS file contributed; shared
        cross-file entities still referenced by other live sources survive
        (the store implements the reference counting). Mirrors the vector
        `delete_by_filter` orphan sweep so re-ingesting an edited file does
        not leave stale graph nodes.
        """
        if not self._graph_enabled:
            return
        try:
            await self.graph_store.delete_by_source([_graph_source_id(repo, path)])
        except Exception as exc:
            _log.warning(
                "graph delete_by_source failed repo=%s path=%s: %s -- proceeding",
                repo, path, exc,
            )

    async def _extract_and_upsert_graph(self, doc, chunks: list) -> None:
        """Extract SRE entities/relationships and upsert to the graph store.

        Runs AFTER chunks are embedded + upserted to the vector store, so a
        graph failure cannot affect the (already-completed) vector index.
        Strictly non-fatal -- every error is logged and swallowed.

        Combines the per-doc extractor output (metadata rules + structured
        files + optional LLM prose) with a metadata sweep over every chunk's
        enriched metadata (so multi-service / per-chunk facets become edges).
        """
        if not self._graph_enabled:
            return
        try:
            entities: dict[str, Entity] = {}
            rels: dict[tuple[str, str, str], Relationship] = {}

            def _absorb(result) -> None:
                for e in getattr(result, "entities", []) or []:
                    entities.setdefault(e.id, e)
                for r in getattr(result, "relationships", []) or []:
                    rels[(r.source_id, r.target_id, r.rel_type)] = r

            # Doc-level extraction (metadata + structured + optional prose).
            _absorb(await self.entity_extractor.extract(doc))

            # Per-chunk metadata sweep -- picks up service/owner_team/env/repo
            # facets the enricher stamped onto individual chunks. Uses the
            # extractor's metadata lane when available; otherwise skipped.
            meta_lane = getattr(self.entity_extractor, "extract_from_metadata", None)
            if callable(meta_lane):
                source_id = _graph_source_id(doc.source.repo, doc.source.path)
                for c in chunks:
                    meta = dict(getattr(c, "metadata", None) or {})
                    meta.setdefault("repo", doc.source.repo)
                    _absorb(await meta_lane(meta, source_chunk_id=source_id))

            if entities:
                await self.graph_store.upsert_entities(list(entities.values()))
            if rels:
                await self.graph_store.upsert_relationships(list(rels.values()))
            _log.info(
                "graph lane repo=%s path=%s entities=%d rels=%d",
                doc.source.repo, doc.source.path, len(entities), len(rels),
            )
        except Exception as exc:
            _log.warning(
                "graph extract/upsert failed repo=%s path=%s: %s -- vector "
                "index unaffected",
                getattr(doc.source, "repo", "?"),
                getattr(doc.source, "path", "?"),
                exc,
            )

    async def index_repo(
        self,
        repo: str,
        branch: str = "main",
        patterns: list[str] | None = None,
    ) -> int:
        patterns = patterns or [
            "**/*.md",
            "**/*.markdown",
            "**/*.tf",
            "**/*.yaml",
            "**/*.yml",
            "**/Dockerfile",
            "**/*.hcl",
            # Helm template partials -- these ARE the chart logic. Missing them
            # made `generic-application` look 10-files-deep when it's really 72.
            "**/*.tpl",
            "**/*.gotmpl",
            # JSON configs (k8s manifests sometimes, dashboard JSONs)
            "**/*.json",
            # Top-level configs we'd want to index even without an extension
            "**/Makefile",
            # Source code -- the organization's stack is Python (Django) + JS/TS
            # (Angular, React, Vue). Without these patterns, queries like
            # "how does fms authenticate?" fall through to README-only context.
            "**/*.py",
            "**/*.pyi",
            "**/*.js", "**/*.jsx", "**/*.mjs", "**/*.cjs",
            "**/*.ts", "**/*.tsx",
            "**/*.vue",
            "**/*.go",
            "**/*.java", "**/*.kt", "**/*.kts",
            "**/*.html", "**/*.htm",
            "**/*.css", "**/*.scss", "**/*.sass", "**/*.less",
            # Shell + env templates (often part of service deployment)
            "**/*.sh", "**/*.bash",
        ]
        # Idempotent -- auto-index path queues earlier; manual / webhook paths
        # need this so start_listing/start_indexing don't no-op.
        indexing_tracker.ensure_queued(repo, branch)
        indexing_tracker.start_listing(repo, branch)
        paths = await self.scm.list_files(repo, branch, patterns)
        # Drop paths that match noise patterns (migrations, lockfiles, etc.).
        before = len(paths)
        paths = [p for p in paths if not _is_excluded(p)]
        if before != len(paths):
            _log.info(
                "excluded %d/%d noise paths repo=%s",
                before - len(paths), before, repo,
            )
        indexing_tracker.start_indexing(repo, branch, len(paths))

        file_parallel = max(
            1, int(os.environ.get("OPSRAG_FILE_PARALLEL", str(_DEFAULT_FILE_PARALLEL)))
        )
        _log.info(
            "index_repo repo=%s branch=%s files=%d file_parallel=%d",
            repo, branch, len(paths), file_parallel,
        )

        # Bounded producer/consumer pipeline.
        #
        # Earlier version did `async for file in ...; tasks.append(create_task(...))`
        # which materialised one Task per file PLUS held the RepoFile content
        # (file body) in memory for every queued file before the semaphore
        # released. For 50K-file repos that's 50K Task objects + 50K RepoFiles
        # parked on the event loop -- wedges the loop's task scheduler and
        # blocks unrelated coroutines (FastAPI request handlers, /health,
        # /indexing/status). Memory plus scheduler overhead per repo grew
        # linearly with file count.
        #
        # New design: a single asyncio.Queue with a small fixed capacity
        # acts as backpressure between the producer (SCM streamer) and a
        # fixed pool of worker coroutines. At most `file_parallel * 2`
        # files exist in memory at once, and exactly `1 + file_parallel`
        # asyncio Tasks ever exist for this repo regardless of file count.
        # Repo size no longer scales the loop's task graph.

        queue_capacity = max(file_parallel * 2, 4)
        # Sentinel placed once per worker tells them to exit cleanly when
        # the producer has streamed every file.
        _SENTINEL: object = object()
        queue: asyncio.Queue = asyncio.Queue(maxsize=queue_capacity)

        total_chunks = 0
        file_count = 0

        async def _process_one(file: RepoFile) -> None:
            nonlocal total_chunks, file_count
            try:
                chunks = await self._process_file(file)
            except Exception as exc:
                _log.warning(
                    "file process failed repo=%s path=%s: %s",
                    file.repo, file.path, exc,
                )
                indexing_tracker.file_skipped(repo, branch)
                return
            total_chunks += chunks
            if chunks > 0:
                indexing_tracker.file_indexed(repo, branch, chunks)
            else:
                indexing_tracker.file_skipped(repo, branch)
            file_count += 1
            if file_count % 50 == 0:
                gc.collect()
                try:
                    import resource
                    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                    _log.info(
                        "memory check repo=%s files=%d rss=%.0fMB chunks=%d",
                        repo, file_count, rss, total_chunks,
                    )
                except Exception:
                    pass

        async def _worker() -> None:
            while True:
                item = await queue.get()
                try:
                    if item is _SENTINEL:
                        return
                    await _process_one(item)
                finally:
                    queue.task_done()

        async def _producer() -> None:
            try:
                async for file in self.scm.get_files_batch(repo, paths, branch):
                    # `put` blocks when the queue is full -> SCM streaming
                    # naturally backpressures. RepoFile content stays out
                    # of memory until a worker is free.
                    await queue.put(file)
            finally:
                # Always send sentinels -- guarantees workers exit even if
                # the SCM stream raises mid-flight.
                for _ in range(file_parallel):
                    await queue.put(_SENTINEL)

        workers = [asyncio.create_task(_worker()) for _ in range(file_parallel)]
        try:
            await _producer()
            await queue.join()
        finally:
            # Workers received sentinels via the producer's finally block;
            # gather them so any worker exception surfaces.
            await asyncio.gather(*workers, return_exceptions=True)

        indexing_tracker.repo_done(repo, branch)
        return total_chunks

    async def index_source(
        self,
        source_type: str,
        scope: str,
        *,
        fetch_concurrency: int = 5,
    ) -> int:
        """Phase 2 -- ingest a non-git source.

        Looks up the configured `SourceProtocol` for `source_type` and
        iterates `(scope)` documents through the same chunker / embedder
        / vector-store path that `index_repo` uses for git. The git
        path stays separate (`index_repo`) because its semantics --
        branches, working trees, file globs -- don't generalize cleanly.

        `scope` is source-specific: a Confluence space key, a Rootly
        project id, a Slack channel name, etc.
        """
        source = self.sources.get(source_type)
        if source is None:
            raise ValueError(
                f"no source provider registered for source_type={source_type!r}; "
                f"registered: {sorted(self.sources)}"
            )

        # The dedup tracker uses (repo, branch, path) as its primary key.
        # Map non-git sources onto it via the convention documented in
        # opsrag.interfaces.source: repo=<source_type>:<scope>,
        # branch=<source_type>, path=<doc_id>.
        repo_key = f"{source_type}:{scope}"
        indexing_tracker.ensure_queued(repo_key, source_type, source_type=source_type)
        indexing_tracker.start_listing(repo_key, source_type)

        # Collect refs first so we can update the indexing tracker with a
        # total count. Sources should yield refs cheaply -- no body fetch
        # in list_documents.
        refs = []
        async for ref in source.list_documents(scope):
            refs.append(ref)
        indexing_tracker.start_indexing(repo_key, source_type, len(refs))
        _log.info(
            "index_source source=%s scope=%s docs=%d concurrency=%d",
            source_type, scope, len(refs), fetch_concurrency,
        )

        sem = asyncio.Semaphore(max(1, fetch_concurrency))
        total_chunks = 0
        chunks_lock = asyncio.Lock()

        # Display-name flag flipped after the first successful fetch so
        # the dashboard can show "slack:#devops" instead of the opaque
        # channel ID. Source-specific: Slack docs carry channel_name,
        # Confluence carries space_key (already readable), git is fine
        # as-is. Only set once per run.
        display_name_set = False

        async def _process_one(ref) -> None:
            nonlocal total_chunks, display_name_set
            async with sem:
                try:
                    doc = await source.fetch_document(ref)
                except Exception as exc:
                    _log.warning(
                        "fetch_document failed source=%s scope=%s doc=%s: %s",
                        source_type, scope, ref.doc_id, exc,
                    )
                    indexing_tracker.file_skipped(repo_key, source_type)
                    return
                if not display_name_set:
                    label = _derive_display_name(source_type, doc.metadata or {})
                    if label:
                        indexing_tracker.set_display_name(repo_key, source_type, label)
                    display_name_set = True
                try:
                    n = await self._process_file(doc)
                except Exception as exc:
                    _log.warning(
                        "process_document failed source=%s scope=%s doc=%s: %s",
                        source_type, scope, ref.doc_id, exc,
                    )
                    indexing_tracker.file_skipped(repo_key, source_type)
                    return
                # Per-page progress: update the dashboard counters before
                # the next page starts. Without this, the UI shows 0/N
                # the entire run and only flips to "done" at the end.
                if n > 0:
                    indexing_tracker.file_indexed(repo_key, source_type, n)
                else:
                    indexing_tracker.file_skipped(repo_key, source_type)
                async with chunks_lock:
                    total_chunks += n

        await asyncio.gather(*(_process_one(ref) for ref in refs))
        indexing_tracker.repo_done(repo_key, source_type)
        return total_chunks

    async def handle_webhook(self, event: WebhookEvent) -> int:
        total = 0
        for path in event.changed_files:
            try:
                file = await self.scm.get_file(event.repo, path, event.branch)
            except Exception as exc:
                _log.warning("webhook fetch failed repo=%s path=%s: %s", event.repo, path, exc)
                continue
            await self.vector_store.delete_by_filter(
                {"repo": event.repo, "source_path": path}
            )
            # Reference-counted graph orphan sweep for the changed file (one
            # of the two delete-sweep sites). Non-fatal; no-op when the graph
            # lane is not wired.
            await self._graph_delete_by_source(event.repo, path)
            total += await self._process_file(file)
        return total

    @staticmethod
    def _source_type_for(file: RepoFile) -> str:
        """Best-effort origin connector for a file/document.

        Non-git sources stamp `source_type` into `RepoFile.metadata` (see
        SourceProtocol.fetch_document). Git files don't, so default to
        "git". Deterministic and side-effect-free -- only used to seed the
        enricher's `source_system` facet.
        """
        meta = getattr(file, "metadata", None) or {}
        st = meta.get("source_type")
        if isinstance(st, str) and st:
            return st
        return "git"

    async def _process_file(self, file: RepoFile) -> int:
        parser = next(
            (p for p in self.parsers if p.supports(file.path, file.content)),
            None,
        )
        if parser is None:
            return 0

        # Step 7: content-hash dedup. Compute sha256 of the file once and
        # ask the tracker if we've already indexed this exact bytes for
        # (repo, branch, path). If yes, skip the entire embed/parse/upsert
        # chain -- the existing Qdrant points are still valid. We still
        # bump last_seen_at so a future deletion sweep knows the file is
        # still present in source.
        content_hash = hashlib.sha256(file.content.encode("utf-8")).hexdigest()
        try:
            if await self.indexed_files.should_skip(
                file.repo, file.branch, file.path, content_hash
            ):
                await self.indexed_files.mark_seen(
                    file.repo, file.branch, [file.path]
                )
                return 0
        except Exception as exc:
            # Tracker failure must never block indexing -- fall through and
            # treat as "needs indexing". The user can investigate Postgres
            # connectivity separately.
            _log.warning(
                "indexed_files.should_skip failed repo=%s path=%s: %s "
                "-- processing anyway",
                file.repo, file.path, exc,
            )
        # parser.parse + chunker.chunk are CPU-bound sync work. Running
        # them directly on the event loop blocks unrelated coroutines --
        # FastAPI request handlers, /health probes -- for hundreds of ms
        # per large file. Offload to the thread pool so the loop stays
        # free to service HTTP traffic while indexing churns.
        try:
            doc = await asyncio.to_thread(parser.parse, file)
        except Exception as exc:
            _log.warning("parse failed repo=%s path=%s: %s", file.repo, file.path, exc)
            return 0

        # Provenance (updated_at/service/url) CENTRALLY so every doc type gets it
        # -- only markdown/helm parsers called apply_provenance, so the bulk of
        # the corpus (code, k8s, terraform, alert, generic) could never be
        # recency-ranked. Applied to doc.metadata before chunking so it
        # propagates onto every chunk. Idempotent (markdown/helm re-apply same
        # values). Non-fatal.
        try:
            from opsrag.ingestion.metadata import apply_provenance
            doc.metadata = doc.metadata or {}
            apply_provenance(doc.metadata, file, source_type=self._source_type_for(file))
        except Exception as exc:
            _log.debug("apply_provenance failed repo=%s path=%s: %s", file.repo, file.path, exc)

        chunks = await asyncio.to_thread(self.chunker.chunk, doc)
        if not chunks:
            return 0

        # Generalized chunk metadata: deterministic, no-LLM enrichment of
        # every chunk's metadata dict (doc_type/environment/tier/tags/
        # language/valid). Runs after chunk() and before embedding so the
        # facets land in the Qdrant payload. Additive only -- never clobbers
        # parser/chunker keys and never touches content, IDs, parent linkage,
        # or vectors, so retrieval is unchanged for chunks that lack any
        # given facet. Non-fatal: enrichment failure must not block indexing.
        source_type = self._source_type_for(file)
        for c in chunks:
            try:
                enrich_metadata(
                    c.metadata,
                    path=c.source_path,
                    text=c.content,
                    source_type=source_type,
                    struct_doc_type=c.doc_type,
                )
            except Exception as exc:
                _log.warning(
                    "metadata enrich failed repo=%s path=%s: %s -- proceeding",
                    file.repo, file.path, exc,
                )

        # Light-graph lane: stamp deterministic `entity_ids` onto each chunk's
        # metadata (so they land in the Qdrant payload) + collect structured
        # edges for the Postgres adjacency table. Runs before embed/upsert so
        # the payload carries entity_ids. No-op + no cost when the light graph
        # isn't wired. Independent of the Neo4j lane.
        _light_edges = await self._attach_entity_ids(chunks, doc, getattr(file, "repo", ""))

        # Step 3: contextual chunking. Selectively prepends a one-sentence
        # context to each child chunk for prose docs (runbooks/postmortems/
        # markdown). Toggled by OPSRAG_CONTEXTUAL_CHUNKING env var. No-op
        # otherwise.
        if contextual.is_enabled() and self.llm is not None:
            try:
                chunks = await contextual.augment_chunks(chunks, doc, self.llm)
            except Exception as exc:
                _log.warning(
                    "contextual augment failed repo=%s path=%s: %s",
                    file.repo, file.path, exc,
                )

        # Orphan sweep -- delete prior (repo, source_path) chunks/entities
        # before upserting. Chunk IDs hash `content[:64]` (parent_child.py),
        # so any edit to the first 64 chars of a chunk creates a new ID; the
        # old vector stays in Qdrant forever, still matching queries. The
        # webhook path (handle_webhook above) already does this delete-sweep;
        # the daily-reindex / `_process_file` path didn't, so stale chunks
        # accumulated for every file ever edited. Bounded cost: one extra
        # call per CHANGED file (hash-mismatch path); unchanged files exit
        # early via `should_skip` above and never reach this point.
        try:
            await self.vector_store.delete_by_filter(
                {"repo": file.repo, "source_path": file.path}
            )
        except Exception as exc:
            _log.warning(
                "pre-upsert delete failed repo=%s path=%s: %s -- proceeding (may orphan)",
                file.repo, file.path, exc,
            )
        # Reference-counted graph orphan sweep (second of the two delete-sweep
        # sites: daily-reindex / `_process_file`). Removes only what this file
        # previously contributed; shared cross-file entities/edges survive.
        # Non-fatal; no-op when the graph lane is not wired.
        await self._graph_delete_by_source(file.repo, file.path)
        # Light-graph per-file sweep: drop the edges THIS file contributed last
        # run before re-upserting below, so renamed/removed entities don't
        # accrete in the Postgres adjacency that drives entity_expand. Non-fatal.
        if self.light_graph is not None:
            try:
                await self.light_graph.delete_by_source(file.repo, file.path)
            except Exception as exc:
                _log.warning(
                    "light-graph pre-upsert delete failed repo=%s path=%s: %s -- proceeding",
                    file.repo, file.path, exc,
                )

        # Embed only the SEARCHABLE chunks. Parents are stored for parent-
        # substitution but are filtered out of every search lane (chunk_type=
        # 'parent'), so paying the embedding API for them spends tokens on
        # vectors no query can ever match. Embed children/standalones; hand the
        # store a None placeholder for parents (pgvector -> NULL embedding,
        # Qdrant -> BM25-only point), keeping the chunks/embeddings lists 1:1.
        searchable = [c for c in chunks if c.chunk_type != "parent"]
        # Embed `embed_content` when set (contextual prefix -> dense lane only),
        # else `content`. BM25/FTS/payload below always use the clean `content`.
        dense = await self.embedder.embed_texts(
            [c.embed_content or c.content for c in searchable]
        )
        _dense_iter = iter(dense)
        embeddings: list[list[float] | None] = [
            None if c.chunk_type == "parent" else next(_dense_iter)
            for c in chunks
        ]

        # Cap upsert payload size; Qdrant accepts large batches but smaller
        # ones surface partial-failure isolation more cleanly.
        upsert_batch = 256
        for i in range(0, len(chunks), upsert_batch):
            await self.vector_store.upsert(
                chunks[i : i + upsert_batch],
                embeddings[i : i + upsert_batch],
            )
        _log.info("indexed repo=%s path=%s chunks=%d", file.repo, file.path, len(chunks))

        # Light-graph edge upsert (after the vector upsert, like the Neo4j lane,
        # so a failure can't affect the completed vector index). Non-fatal.
        if self.light_graph is not None and _light_edges:
            try:
                await self.light_graph.upsert_edges(
                    _light_edges,
                    repo=getattr(file, "repo", ""),
                    source_path=getattr(file, "path", ""),
                )
            except Exception as exc:
                _log.warning(
                    "light-graph edge upsert failed repo=%s: %s -- proceeding",
                    getattr(file, "repo", "?"), exc,
                )

        # P3 -- dual-write code chunks into the code collection. Only the
        # code DocTypes get dual-written; prose/config chunks stay in the
        # main collection only. Failure here is non-fatal -- main collection
        # already has the chunks, so retrieval still works; we just lose
        # the code lane for this file.
        await self._maybe_dual_write_code(file, chunks)

        # Graph lane (re-activated): extract SRE entities/relationships from
        # this doc's metadata + prose and upsert to the graph store. Runs
        # AFTER the vector upsert so a graph failure leaves the vector index
        # intact. Strictly non-fatal -- swallowed inside the helper. No-op
        # when the graph lane is not wired (default deployment).
        await self._extract_and_upsert_graph(doc, chunks)

        # Step 7: record the (repo, branch, path) -> content_hash entry now
        # that Qdrant has the chunks. Subsequent reindex runs can short-
        # circuit. Failure here is non-fatal -- the file IS indexed; we
        # just won't dedup it next run.
        try:
            await self.indexed_files.record(
                file.repo, file.branch, file.path, content_hash, len(chunks)
            )
        except Exception as exc:
            _log.warning(
                "indexed_files.record failed repo=%s path=%s: %s",
                file.repo, file.path, exc,
            )

        return len(chunks)

    async def _maybe_dual_write_code(self, file, chunks: list) -> None:
        """P3 -- additionally embed + upsert code chunks into the code collection.

        Skipped entirely if `code_embedder` or `code_vector_store` is None.
        Skipped when no chunks have a code DocType. Failure is non-fatal:
        the main collection already received the chunks, so retrieval
        continues to work via the existing lanes.
        """
        if self.code_embedder is None or self.code_vector_store is None:
            return
        code_chunks = [c for c in chunks if c.doc_type in _CODE_DOC_TYPES]
        if not code_chunks:
            return
        # Orphan sweep on the code collection to mirror the main-collection
        # sweep above. Edits to a file's first 64 chars create new chunk IDs;
        # without this, old code-collection vectors would linger.
        try:
            await self.code_vector_store.delete_by_filter(
                {"repo": file.repo, "source_path": file.path}
            )
        except Exception as exc:
            _log.warning(
                "code-collection delete_by_filter failed repo=%s path=%s: %s -- proceeding",
                file.repo, file.path, exc,
            )
        try:
            # Same parent-skip as the main lane: code parents are excluded from
            # code search, so don't spend the code embedder on them.
            _code_searchable = [c for c in code_chunks if c.chunk_type != "parent"]
            _code_dense = await self.code_embedder.embed_texts(
                [c.embed_content or c.content for c in _code_searchable]
            )
            _code_iter = iter(_code_dense)
            code_embeddings: list[list[float] | None] = [
                None if c.chunk_type == "parent" else next(_code_iter)
                for c in code_chunks
            ]
        except Exception as exc:
            _log.warning(
                "code embedder failed repo=%s path=%s: %s -- skipping code lane for this file",
                file.repo, file.path, exc,
            )
            return
        upsert_batch = 256
        for i in range(0, len(code_chunks), upsert_batch):
            try:
                await self.code_vector_store.upsert(
                    code_chunks[i : i + upsert_batch],
                    code_embeddings[i : i + upsert_batch],
                )
            except Exception as exc:
                _log.warning(
                    "code-collection upsert failed repo=%s path=%s: %s",
                    file.repo, file.path, exc,
                )
                return
        _log.info(
            "indexed (code lane) repo=%s path=%s code_chunks=%d",
            file.repo, file.path, len(code_chunks),
        )
