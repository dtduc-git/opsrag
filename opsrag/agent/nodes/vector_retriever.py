"""Vector retriever node.

Boosts on top of plain vector search:

1. **Repo scoping with hard/soft modes** -- if the user names a repo, we
   either *hard filter* (Qdrant payload filter) or *soft boost* (no
   filter, but pull more candidates and let the reranker sort). Hard
   when the query unambiguously names the repo (full path, or
   `in/from <slug>`, or `<slug> repo`); soft when it's a bare-slug
   mention that could legitimately span multiple repos. Plural intent
   ("all repos", "which repositories") disables scoping entirely.

2. **Listing-intent top_k boost** -- for "what's the structure / list of
   files / contents of" queries, pull 3x normal top_k.

3. **Structural file enumeration** -- for listing queries that name a
   directory path (e.g. "files in projects/staging/apps/qa-runner"), the right
   primitive is enumerate-distinct-paths, NOT semantic search. Vector
   similarity only retrieves chunks whose CONTENT matches the query;
   files whose content doesn't say "list" or "directory" are invisible
   to it. We use Qdrant scroll to enumerate files structurally and
   synthesize a directory-listing chunk into the LLM context.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time

_log = logging.getLogger("opsrag.vector_retriever")

from opsrag.interfaces.chunker import Chunk
from opsrag.interfaces.embedder import EmbeddingProvider
from opsrag.interfaces.observability import ObservabilityProvider
from opsrag.interfaces.parser import DocType
from opsrag.interfaces.vectorstore import SearchResult, VectorStore
from opsrag.vectorstores.priority import priority_multiplier
from opsrag.vectorstores.rrf import rrf_merge_pools

# Minimum candidate pool handed to the reranker, independent of the answer
# top_k. Reranking is the highest-ROI lever but can only re-order what it's
# given; a wide pool lets the cross-encoder promote a doc the bi-encoder ranked
# deep. Truncation back to the output size happens in the rerank node.
_RERANK_CANDIDATE_POOL = 40

# Match GitLab/GitHub-style repo paths: owner/repo or owner/group/.../repo.
_REPO_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_/.-])([A-Za-z0-9_-]+(?:/[A-Za-z0-9_.-]+){1,4})"
)
# Match a single repo-ish slug ("gitops-pipeline-templates"). Used as a
# secondary pass when nothing with a slash matched.
_SLUG_RE = re.compile(r"[A-Za-z][A-Za-z0-9._-]*")
# Drop common English/operator words so we don't match "structure" against a repo basename.
_STOPWORDS = {
    "structure", "files", "file", "details", "purpose", "more", "tell",
    "give", "show", "list", "what", "where", "which", "this", "that",
    "the", "for", "all", "and", "with", "from", "into", "about", "repo",
    "repository", "directory", "service", "services", "config", "yaml",
    "yml", "production", "prod", "staging", "dev", "main", "master",
    "branch", "please", "pls", "can", "you", "me", "i", "of", "in",
    "at", "to", "is", "are", "do", "be", "have", "has", "ci", "cd",
}


_LISTING_HINTS = (
    "all the files", "all files", "list of files", "structure of",
    "what files", "what's in", "what is in", "contents of", "files in",
    "directory structure", "tree of", "every file", "each file",
    "how many files", "how many distinct files", "list all", "list the",
    "files of", "files for", "files under",
)


def _build_token_index(known_repos: list[str]) -> dict[str, set[str]]:
    """Map every token derivable from a repo path to the repos containing it.

    Tokens come from: full path segments + dash-split sub-tokens. Used so
    "terraform" -> terraform-modules repo (path segment), "monorepo" ->
    gitops (dash split), "notes" -> saas/acme-notes-be/backend.
    Tokens of 3+ chars are indexed; uniqueness check at lookup time
    prevents ambiguous matches.
    """
    out: dict[str, set[str]] = {}
    for r in known_repos:
        for segment in r.split("/"):
            if len(segment) > 2:
                out.setdefault(segment.lower(), set()).add(r)
            for piece in segment.split("-"):
                if len(piece) > 2:
                    out.setdefault(piece.lower(), set()).add(r)
    return out


def _detect_repo(query: str, known_repos: list[str]) -> str | None:
    """Return the longest known repo path mentioned in the query, else None.

    Detection layers (in order):
    1. Full-path candidates ("devops/foo/bar") -- exact known-repo match wins.
    2. Path-suffix candidates -- query mentions a repo's last segment.
    3. Bare-slug candidates against any unique repo path component (handles
       "terraform" -> "devops/terraform-modules/acme-tf-state/terraform"
       and "monorepo" -> "devops/gitops").
    """
    if not known_repos:
        return None

    # Pass 1: full-path candidates ("devops/foo" or "devops/foo/bar").
    path_candidates = _REPO_PATH_RE.findall(query)
    exact = sorted(
        (c for c in path_candidates if c in known_repos),
        key=len, reverse=True,
    )
    if exact:
        return exact[0]
    for c in sorted(path_candidates, key=len, reverse=True):
        for r in known_repos:
            if r.endswith("/" + c) or r.split("/")[-1] == c:
                return r

    # Pass 2: bare-slug match against any unique token of a known repo.
    # 3+ chars covers short repo basenames (fms, sms, car); uniqueness
    # check below filters out ambiguous matches like "saas" or "dm".
    token_to_repos = _build_token_index(known_repos)
    slugs = [
        s for s in _SLUG_RE.findall(query)
        if len(s) > 2 and s.lower() not in _STOPWORDS
    ]
    # Try longest slugs first so "terraform2.0" wins over "terraform" if both
    # appear in the same query.
    for s in sorted(slugs, key=len, reverse=True):
        repos = token_to_repos.get(s.lower())
        # Only scope when the token is unambiguous -- maps to exactly one repo.
        if repos and len(repos) == 1:
            return next(iter(repos))
    return None


def _is_listing_intent(query: str) -> bool:
    low = query.lower()
    return any(h in low for h in _LISTING_HINTS)


# Synthesis intent -- queries that need 2+ docs to answer well.
# Comparison / relationship / cross-reference language. We boost top_k
# for these so the reranker has both anchors to pick from. Defensive
# fix for multi_doc_synthesis recall plateauing at ~0.45 in eval.
_SYNTHESIS_PATTERNS = re.compile(
    r"\b("
    r"compare|comparison|relationship|relate|relates|related"
    r"|difference\s+between|differences?|vs\.?|versus"
    r"|how\s+(?:does|do)\s+\w+\s+(?:relate|interact|connect)"
    r"|decide\s+(?:whether|between)|choose\s+between"
    r"|specific\s+to|distinguish|contrast"
    r")\b",
    re.IGNORECASE,
)


def _is_synthesis_intent(query: str) -> bool:
    """True when the query asks to compare or relate multiple docs.

    Examples that match: "compare A against B", "relationship between
    cd.yaml and dynamic-pipeline", "what's specific to acme-notes-be vs the
    platform-wide one", "how does generic-pipeline decide between X
    and Y".
    """
    return bool(_SYNTHESIS_PATTERNS.search(query))


# Plural / cross-repo intent signals -- when present, scoping to one
# repo would actively hide what the user asked for.
_PLURAL_PATTERNS = re.compile(
    r"\b("
    r"all\s+(?:the\s+)?(?:repos?|repositories|services|projects)"
    r"|which\s+(?:repos?|repositories|services|projects)"
    r"|every\s+(?:repo|repository|service)"
    r"|each\s+(?:repo|repository|service)"
    r"|across\s+(?:repos?|repositories|services)"
    r"|(?:list|enumerate)\s+all\s+(?:repos?|repositories|services|projects)"
    r"|repos?\s+that\s+(?:has|have|contain|configure)"
    r"|repositories\s+that\s+(?:has|have|contain|configure)"
    r")\b",
    re.IGNORECASE,
)


def _has_plural_repo_intent(query: str) -> bool:
    """True when the query asks for cross-repo coverage.

    Examples that match: "all repositories that have config of X",
    "which repos handle Y", "list all services with Z", "across repos".
    Examples that don't: "what's in repo X", "tell me about <slug>".
    """
    return bool(_PLURAL_PATTERNS.search(query))


# Slug worth fanning out on -- must contain a dash (so "acme-notes-be", "gateway-lens",
# "kong-ingress" qualify; "config", "service", "argocd" don't unless they're
# known repo tokens).
_DASHED_SLUG_RE = re.compile(r"\b([a-z][a-z0-9]*(?:-[a-z0-9]+)+)\b", re.IGNORECASE)

# Filenames literally mentioned in the query, e.g. "cd.yaml", "preview.yaml",
# "main.tf", "Chart.yaml", "incident-profile.md". These deserve a direct
# path-pattern lookup because:
#   1. Single-word stems ("cd", "preview", "common") don't qualify for the
#      dashed-slug fanout above, so embedding-only retrieval often misses
#      them when the query is comparing two specific files.
#   2. The user is naming a file ON PURPOSE -- they want THAT FILE, not
#      semantic neighbors. Treat it as a near-exact lookup.
_FILENAME_RE = re.compile(
    r"\b([A-Za-z0-9_.-]+\.(?:ya?ml|tf|md|py|json|sh|hcl|toml))\b"
)


def _extract_filenames(query: str) -> list[str]:
    """Filenames mentioned literally. Deduped, capped at 5 to keep the
    fan-out budget bounded."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _FILENAME_RE.findall(query):
        low = m.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(low)
        if len(out) >= 5:
            break
    return out


def _extract_service_slugs(query: str, known_repos: list[str]) -> list[str]:
    """Slugs likely to be service / module names worth a path-pattern fan-out.

    Two sources, deduped:
    1. Dashed compound slugs in the query (e.g. "acme-notes-be", "gateway-lens")
       -- these are almost always service names; very low false-positive rate.
    2. Single-token bare slugs that uniquely match a known repo segment
       (e.g. "fms", "argocd", "terraform" if the index knows that token).

    Excluded: stopwords, single-char, generic words. Caller will pass each
    slug to vector_store.search_by_text to surface chunks across all repos
    whose content (and usually source_path) literally contains the slug.
    """
    out: list[str] = []
    seen: set[str] = set()

    # 1. Dashed compounds.
    for m in _DASHED_SLUG_RE.findall(query):
        s = m.lower()
        if s in seen or s in _STOPWORDS:
            continue
        seen.add(s)
        out.append(s)

    # 2. Bare slugs that map uniquely to a known repo.
    if known_repos:
        token_to_repos = _build_token_index(known_repos)
        for s in _SLUG_RE.findall(query):
            low = s.lower()
            if low in seen or low in _STOPWORDS or len(low) < 3:
                continue
            repos = token_to_repos.get(low)
            # Unique token = service-like. Skip ambiguous tokens like "saas".
            if repos and len(repos) == 1:
                seen.add(low)
                out.append(low)

    return out


def _slug_tokens(repo: str) -> list[str]:
    """All discriminative tokens of a repo path: full segments + dash splits."""
    out: list[str] = []
    for segment in repo.split("/"):
        if len(segment) > 2:
            out.append(segment.lower())
        for piece in segment.split("-"):
            if len(piece) > 2:
                out.append(piece.lower())
    return out


def _should_hard_filter_repo(query: str, repo: str) -> bool:
    """True when the query unambiguously scopes to this repo.

    Hard cases:
      - Full repo path appears verbatim (e.g. "devops/foo/bar").
      - "in <slug>" / "from <slug>" / "<slug> repo[sitory]" qualifier
        where <slug> is a discriminative token of `repo`.

    Soft case (return False): bare slug mention with no qualifier
    ("tell me about acme-notes-be"). Caller should retrieve without filter
    and let the reranker decide.
    """
    if not repo:
        return False
    low = query.lower()
    # Full path verbatim -> unambiguous.
    if repo.lower() in low:
        return True
    # "in <slug>" / "from <slug>" / "<slug> repo[sitory]" qualifier.
    qualifier = re.compile(
        r"\b(?:in|from|inside|within)\s+(?P<slug>[a-z0-9][a-z0-9._-]+)"
        r"|(?P<slug2>[a-z0-9][a-z0-9._-]+)\s+(?:repo|repository)\b",
        re.IGNORECASE,
    )
    repo_tokens = set(_slug_tokens(repo))
    for m in qualifier.finditer(low):
        slug = (m.group("slug") or m.group("slug2") or "").lower()
        if slug and slug in repo_tokens:
            return True
    return False


# Detect a directory-like path (e.g. "projects/staging/apps/qa-runner") that
# isn't a known repo. At least 2 segments, no leading http/dot.
_PATH_RE = re.compile(r"(?<![A-Za-z0-9_/.-])([a-zA-Z][a-zA-Z0-9._-]*(?:/[a-zA-Z0-9._-]+){1,15})")


def _detect_path(query: str, exclude_repo: str | None = None) -> str | None:
    """Find a directory-like path in the query, distinct from the repo.

    Returns the longest path that:
    - Has at least 2 path segments (e.g. "a/b" minimum).
    - Is NOT a known repo path or its prefix (we already detected those).
    - Doesn't look like a URL or a filename with extension.
    """
    candidates = _PATH_RE.findall(query)
    repo_str = exclude_repo or ""
    out: list[str] = []
    for c in candidates:
        c = c.rstrip("/")
        if c.startswith(("http://", "https://")):
            continue
        if exclude_repo and (c == repo_str or repo_str.startswith(c + "/")):
            continue
        # Strip the repo prefix if present so "devops/repo/path/to/dir" -> "path/to/dir"
        if repo_str and c.startswith(repo_str + "/"):
            c = c[len(repo_str) + 1 :]
        if "/" not in c:
            continue
        out.append(c)
    if not out:
        return None
    return max(out, key=len)


def vector_retrieve_node(
    vector_store: VectorStore,
    embedder: EmbeddingProvider,
    observability: ObservabilityProvider,
    top_k: int = 5,
    known_repos: list[str] | None = None,
    code_embedder: EmbeddingProvider | None = None,
    code_store: VectorStore | None = None,
):
    known = known_repos or []

    # The main retrieval lane uses hybrid_search (dense + BM25 RRF) rather than
    # dense-only search, so exact identifier / error-string / config-key matches
    # -- which dense embeddings blur -- are recalled via the lexical lane. Code
    # and ops queries lean heavily on this. The optional code lane adds a 4th
    # RRF lane over the `opsrag_code` collection, but only Qdrant's
    # hybrid_search accepts the code_embedding / code_store kwargs (pgvector's
    # does not). Feature-detect once so any store without those kwargs degrades
    # cleanly to dense+BM25 instead of raising TypeError.
    _code_lane_supported = False
    try:
        _hs_params = inspect.signature(vector_store.hybrid_search).parameters
        _code_lane_supported = (
            "code_embedding" in _hs_params and "code_store" in _hs_params
        )
    except (ValueError, TypeError):  # C-impl / builtin without an introspectable signature
        _code_lane_supported = False
    _code_lane_on = bool(
        code_embedder is not None and code_store is not None and _code_lane_supported
    )

    async def _retrieve(state: dict) -> dict:
        query = state["query"]
        plural = _has_plural_repo_intent(query)
        # Plural intent always wins -- disables single-repo scoping entirely.
        repo = None if plural else _detect_repo(query, known)
        # Decide filter mode: hard (Qdrant filter) vs soft (broader recall,
        # let reranker sort). See module docstring.
        hard_filter = bool(repo) and _should_hard_filter_repo(query, repo)
        listing = _is_listing_intent(query)
        path_prefix = _detect_path(query, exclude_repo=repo) if listing else None

        synthesis = _is_synthesis_intent(query)
        scoped_top_k = top_k
        if listing:
            scoped_top_k = min(top_k * 3, 50)
        elif hard_filter:
            scoped_top_k = min(top_k * 2, 30)
        elif synthesis:
            # Cross-doc queries need both anchors in the rerank pool.
            # Default top_k=5 is too narrow -- bump to listing's 3x.
            scoped_top_k = min(top_k * 3, 50)
        elif repo:
            # Soft mode: fetch wider so the reranker has alternatives across
            # repos that mention the slug. Same factor as listing intent.
            scoped_top_k = min(top_k * 3, 50)

        # Candidate-pool floor for reranking. A cross-encoder can only rescue a
        # doc the bi-encoder ranked low if that doc is in the pool -- a "vanilla"
        # query with no intent signal otherwise fetched only `top_k` (~10),
        # making the (now default-on) reranker nearly inert. The rerank node
        # truncates back to the output size and grading runs AFTER rerank, so a
        # wide retrieval pool costs an extra cross-encoder pass, NOT extra
        # per-doc LLM grader calls.
        scoped_top_k = max(scoped_top_k, _RERANK_CANDIDATE_POOL)

        filters = {"repo": repo} if hard_filter else None

        # Path-pattern fan-out: when the query mentions a service slug
        # (e.g. "acme-notes-be", "gateway-lens"), embedding alone often misses
        # config/values files whose embedding is dominated by YAML structure
        # rather than the service name. Run a parallel text-match scroll
        # on `content` for each detected slug -- this surfaces files that
        # literally mention the slug across all repos. Skip when hard-
        # filtering (user explicitly named one repo) since we'd just be
        # subsampling the same scope.
        fanout_slugs: list[str] = (
            [] if hard_filter else _extract_service_slugs(query, known)
        )
        # Cap the fan-out so a query mentioning many slugs doesn't explode.
        fanout_slugs = fanout_slugs[:3]

        # Filename fanout: literal `*.yaml/*.tf/*.md/...` mentioned in the
        # query. Single-word filenames (cd.yaml, preview.yaml) don't pass
        # the dashed-slug filter, so embedding alone misses them when the
        # user is comparing two specific files. Run a parallel
        # source_path text-match so the file itself reaches the rerank
        # pool. Skip when hard-filtering by repo (same scope already).
        fanout_filenames: list[str] = (
            [] if hard_filter else _extract_filenames(query)
        )

        start = time.perf_counter()
        # T1.5 HyDE: if the upstream node generated a hypothetical
        # answer, embed THAT instead of the raw query. Closes
        # vocabulary gaps between user phrasing and our corpus (e.g.
        # "add a secret" vs the docs' "ExternalSecret + envFrom").
        # We still keep `query` for filter/scoping detection, repo
        # heuristics, and listing intent above -- HyDE only changes
        # the embedding fed to the vector index.
        hyde_text = state.get("hyde_text")
        embed_target = hyde_text if hyde_text else query
        embedding = await embedder.embed_query(embed_target)

        # Code lane: embed the query with the code-specific embedder so
        # hybrid_search can fuse a 4th RRF lane over the `opsrag_code`
        # collection. Use the RAW query, NOT the HyDE expansion: the code
        # embedder runs CODE_RETRIEVAL_QUERY, which expects identifier/code-like
        # query text -- a Flash-generated prose hypothetical ("the service uses
        # a middleware that...") is the opposite of what that lane wants and
        # dilutes exactly the symbol queries the code lane exists to win.
        # Best-effort -- a code-embedder hiccup must never sink the main
        # retrieval, so fall back to dense+BM25 on error.
        code_embedding = None
        if _code_lane_on:
            try:
                code_embedding = await code_embedder.embed_query(query)
            except Exception:
                _log.warning(
                    "code-lane query embed failed; proceeding dense+BM25 only",
                    exc_info=True,
                )
                code_embedding = None

        # Main retrieval lane: hybrid (dense + BM25 RRF). Dense is fed the
        # HyDE-expanded embedding (bridges user-phrasing vs corpus vocabulary);
        # BM25 is fed the RAW user query -- exact tokens (function names, error
        # strings, config keys) must match literally, and HyDE prose would
        # dilute that lexical signal.
        hs_kwargs: dict = dict(
            embedding=embedding,
            query_text=query,
            top_k=scoped_top_k,
            filters=filters,
        )
        if code_embedding is not None:
            hs_kwargs["code_embedding"] = code_embedding
            hs_kwargs["code_store"] = code_store
        # Run the main hybrid search and per-slug / per-filename scrolls in parallel.
        main_task = vector_store.hybrid_search(**hs_kwargs)
        # Stage 1: vector search + slug fanout in parallel. Filename
        # fanout deliberately waits for the vector results -- we use the
        # repos from the top vector hits to *target* the filename
        # scroll, so we don't drown in 377 unrelated `preview.yaml`
        # matches across the whole index.
        slug_fanout_tasks: list = [
            vector_store.search_by_text(text=slug, top_k=top_k * 2)
            for slug in fanout_slugs
            if hasattr(vector_store, "search_by_text")
        ]
        n_slug = len(slug_fanout_tasks)
        stage1 = await asyncio.gather(
            main_task, *slug_fanout_tasks, return_exceptions=True,
        )
        results = stage1[0] if not isinstance(stage1[0], Exception) else []

        slug_results: list = []
        for r in stage1[1 : 1 + n_slug]:
            if not isinstance(r, Exception):
                slug_results.extend(r)

        # Stage 2: filename fanout, scoped to the repos that vector
        # search surfaced. When a user names multiple files in one
        # query they almost always live in the same repo (e.g.
        # "generic-pipeline.yaml decides between preview.yaml and
        # common.yaml" -- all in `devops/gitops-pipeline-templates`).
        # Issuing one targeted scroll per (filename, repo) pair finds
        # the true match deterministically. We also keep one
        # unfiltered fallback scroll per filename so queries whose
        # anchor isn't already in the vector hits still surface
        # something.
        # Rank context repos by frequency in vector hits -- repos that
        # appear in MORE top-vector chunks are more likely to be the
        # relevant ones. set->list[:N] is non-deterministic in Python
        # and was silently dropping the right repo.
        from collections import Counter as _Counter
        repo_counts: _Counter[str] = _Counter()
        for r in results:
            # `repo` is a top-level Chunk field, NOT a nested metadata key --
            # reading metadata.get("repo") returned nothing, so repo_counts was
            # always empty and the targeted per-repo filename scroll below was
            # dead (only the unfiltered top-3 fallback ever ran).
            md = r.chunk.metadata or {}
            repo_name = (
                getattr(r.chunk, "repo", "") or md.get("repo") or md.get("repository") or ""
            )
            if repo_name:
                repo_counts[repo_name] += 1
        # All distinct repos from the vector pool (no truncation --
        # `MatchAny` filter below makes this a single scroll regardless
        # of repo count).
        context_repos: set[str] = set(repo_counts)
        ranked_repos = [r for r, _ in repo_counts.most_common()]

        filename_results: list = []
        if fanout_filenames and hasattr(vector_store, "search_by_path"):
            filename_fanout_tasks: list = []
            # ONE targeted scroll per filename, filtered to ALL
            # context repos via MatchAny. This finds the right file
            # deterministically regardless of how many repos are in
            # the vector hit pool. Top-k generous so a multi-chunk
            # file (parent + children) all surfaces.
            for fname in fanout_filenames:
                if ranked_repos:
                    filename_fanout_tasks.append(
                        vector_store.search_by_path(
                            path_text=fname,
                            top_k=10,
                            filters={"repo": ranked_repos},
                        )
                    )
                # Unfiltered fallback for orphan-filename queries
                # where the anchor file isn't in the vector hits yet.
                filename_fanout_tasks.append(
                    vector_store.search_by_path(path_text=fname, top_k=3)
                )
            stage2 = await asyncio.gather(*filename_fanout_tasks, return_exceptions=True)
            for raw in stage2:
                if isinstance(raw, Exception):
                    continue
                filename_results.extend(raw)

        fanout_results = slug_results + filename_results

        # Merge the main hybrid pool with the slug/filename fanout pools via RRF.
        # The previous raw-score max() was wrong: the main pool carries RRF-fused
        # scores (~0.01-0.02) while the fanout pools carry raw cosine (~0.9), so a
        # fanout hit ALWAYS won a collision and could outrank a strong main hit.
        # Cross-pool RRF puts every pool on the same rank scale; chunks seen in
        # multiple pools accumulate (consensus). The main pool is weighted higher
        # so the user's literal query stays primary and the fanout only augments.
        # top_k = union size so no candidate is dropped before grading/rerank.
        if fanout_results:
            results = rrf_merge_pools(
                [results, slug_results, filename_results],
                top_k=len(results) + len(fanout_results),
                pool_weights=[1.0, 0.5, 0.5],
            )
            # rrf_merge_pools re-derives scores from rank and DISCARDS the
            # authoritative-content priority boost the vector store applied
            # (SRE-KB / architecture / user-correction). Re-apply it here so a
            # fanout query doesn't silently lose the boost. The priority tag was
            # carried onto chunk.metadata by the store's _hit_to_result.
            reboosted: list = []
            for sr in results:
                mult = priority_multiplier((sr.chunk.metadata or {}).get("priority"))
                reboosted.append(
                    sr if mult == 1.0
                    else SearchResult(chunk=sr.chunk, score=sr.score * mult,
                                      distance_metric=sr.distance_metric)
                )
            reboosted.sort(key=lambda s: s.score, reverse=True)
            results = reboosted

        latency_ms = (time.perf_counter() - start) * 1000

        main_count = len(stage1[0]) if not isinstance(stage1[0], Exception) else 0
        slug_count = len(slug_results)
        filename_count = len(filename_results)
        _log.info(
            "retrieve repo=%s mode=%s plural=%s slugs=%s filenames=%s "
            "main=%d slug=%d filename=%d merged=%d ctx_repos=%d",
            repo,
            "hard" if hard_filter else ("soft" if repo else None),
            plural,
            fanout_slugs,
            fanout_filenames,
            main_count, slug_count, filename_count,
            len(results),
            len(context_repos),
        )

        await observability.log_retrieval(
            query=query,
            results=results,
            graph_results=None,
            latency_ms=latency_ms,
            node_name="vector_retrieve",
        )

        chunks = [r.chunk for r in results]

        # Structural augmentation: when query asks to list files in a
        # directory, vector similarity alone misses files whose content
        # doesn't say the directory name. Enumerate distinct file paths
        # via scroll and prepend a synthetic directory-listing chunk.
        if listing and (path_prefix or repo) and hasattr(vector_store, "list_files"):
            try:
                _LIST_LIMIT = 500
                files, total = await vector_store.list_files(
                    repo=repo, path_prefix=path_prefix, limit=_LIST_LIMIT,
                )
                if files:
                    scope_label = repo or "all repos"
                    if path_prefix:
                        scope_label = f"{scope_label} - {path_prefix}/"
                    truncated_note = (
                        f" (showing first {_LIST_LIMIT}, full count above is exact)"
                        if total > _LIST_LIMIT else ""
                    )
                    listing_md = (
                        f"# Directory listing -- {scope_label}\n\n"
                        f"**Total distinct files:** {total}{truncated_note}\n\n"
                        + "\n".join(f"- {f}" for f in files)
                    )
                    synthetic = Chunk(
                        id=f"synthetic:listing:{repo or '*'}:{path_prefix or '*'}",
                        content=listing_md,
                        doc_type=DocType.GENERIC_MARKDOWN,
                        source_path=f"<dir-listing:{path_prefix or repo}>",
                        repo=repo or "",
                        metadata={
                            "synthetic": True,
                            "listing_total": total,
                            "listing_shown": len(files),
                        },
                        parent_chunk_id=None,
                        chunk_type="parent",
                        token_count=max(1, len(listing_md) // 4),
                    )
                    chunks.insert(0, synthetic)
            except Exception:
                # Listing augmentation is best-effort; vector chunks still
                # provide answer fallback.
                pass

        return {
            "retrieved_chunks": chunks,
            "sources_searched": (
                ["vector", "bm25"]
                + (["code"] if code_embedding is not None else [])
                + (["text-fanout"] if fanout_slugs else [])
            ),
            "current_step": "retrieved",
            "scoped_repo": repo,
            "scoped_repo_mode": "hard" if hard_filter else ("soft" if repo else None),
            "plural_repo_intent": plural,
            "fanout_slugs": fanout_slugs,
            "fanout_filenames": fanout_filenames,
            "listing_intent": listing,
            "synthesis_intent": synthesis,
            "scoped_path": path_prefix,
        }

    return _retrieve
