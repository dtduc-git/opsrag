"""Aggregate retrieved chunk paths into a parent/sub-module directory tree.

Motivation: queries like "show me the Terraform modules in X" want a
directory-listing answer (parent-modules + their submodules), not a
random sample of 8 files. The retriever gives us individual chunks at
file granularity, which the LLM then lists verbatim -- producing
"here are 8 files from this repo" instead of "here are the 23 module
categories in this repo".

This module builds a 2-level tree from a list of chunks' `source_path`
fields, focused on a configurable "pivot directory" (default
`modules/`). It returns a compact textual summary the generator can
inject into the LLM context so the model can ground a structural
answer in the *full* set of categories present in retrieval, not just
the 8 best-scoring children.

Example input chunks' source_paths:
    modules/gcp/main.tf
    modules/gcp/iam/main.tf
    modules/cloudflare/zone/main.tf
    modules/pagerduty/env/main.tf
    modules/alloydb/cluster/inputs.tf
    projects/prod/gcp/iam_variables.tf

Example output for pivot='modules/':
    Directory tree under `modules/` derived from N retrieved sources:
    - `alloydb/`  (subdirs: cluster)
    - `cloudflare/`  (subdirs: zone)
    - `gcp/`  (subdirs: iam)
    - `pagerduty/`  (subdirs: env)

The N count tells the LLM how broad the evidence is. The tree shows
all top-level categories WITHOUT relying on the chunk ranker happening
to pick at least one chunk per category.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from opsrag.interfaces.chunker import Chunk

# Pivot directories the agent commonly wants enumerated. Order matters
# -- first match wins per chunk so a `modules/foo/main.tf` is grouped
# under `modules/` and not `foo/`.
_PIVOT_DIRS: tuple[str, ...] = (
    "modules/",
    "projects/",
    "envs/",
    "environments/",
    "charts/",
    "values/",
    "apps/",
    "services/",
    "playbooks/",
    "ansible/",
    "terraform/",
)

# When to bother emitting a tree summary: only if at least this many
# chunks fall under the same pivot AND there are at least this many
# distinct top-level subdirs. Otherwise the LLM is better off with the
# raw chunks alone.
_MIN_CHUNKS_FOR_SUMMARY = 4
_MIN_DISTINCT_TOPS_FOR_SUMMARY = 3
# Cap rendered subdirs per top so the prompt doesn't blow up on big
# repos. Truncated list ends in `...`.
# 100 because gitops-monorepo's `charts/cluster/` has 61 children and
# the WHOLE point of the tree is to enumerate them; cutting to 8 makes
# the agent answer worse than not having the tree at all.
_MAX_SUBDIRS_PER_TOP = 100


def _segments(path: str) -> list[str]:
    return [s for s in path.split("/") if s]


def _find_pivot(path: str) -> tuple[str, list[str]] | None:
    """Return (pivot, rest_segments) if `path` lies under any known pivot,
    where rest_segments is everything AFTER the pivot directory.

    `modules/gcp/iam/main.tf` -> ("modules/", ["gcp", "iam", "main.tf"])
    `projects/prod/gcp/foo.tf` -> ("projects/", ["prod", "gcp", "foo.tf"])
    `README.md` -> None
    """
    p = path
    if not p:
        return None
    for piv in _PIVOT_DIRS:
        # Match either at the very start, or after a "/" -- so a nested
        # `apps/foo/modules/bar/...` still groups bar under modules.
        idx = p.find(piv)
        if idx == 0 or (idx > 0 and p[idx - 1] == "/"):
            after = p[idx + len(piv):]
            rest = _segments(after)
            if rest:
                return piv, rest
    return None


def _pivot_hint_from_query(query: str) -> str | None:
    """If the user's query mentions a pivot category by name (e.g. "list
    all charts", "show me the modules", "what envs do we have"), return
    the matching pivot dir. Used to pick the RIGHT pivot when the
    target repo has multiple pivots indexed (e.g. gitops-monorepo has
    both `charts/` AND `envs/`, the user asking about "charts" wants
    `charts/` not the first alphabetical hit)."""
    if not query:
        return None
    q = query.lower()
    # Order matters -- match the most specific noun first.
    keywords = (
        ("chart", "charts/"),
        ("module", "modules/"),
        ("project", "projects/"),
        ("environment", "environments/"),
        ("env", "envs/"),
        ("app", "apps/"),
        ("service", "services/"),
        ("playbook", "playbooks/"),
        ("ansible", "ansible/"),
        ("value", "values/"),
    )
    for kw, piv in keywords:
        if kw in q and piv in _PIVOT_DIRS:
            return piv
    return None


async def build_path_tree_summary_async(
    chunks: Iterable[Chunk],
    *,
    target_repo: str | None = None,
    vector_store=None,
    query: str | None = None,
) -> str:
    """Like `build_path_tree_summary`, but when `vector_store` is given
    AND `target_repo` is set, enumerate the COMPLETE list of distinct
    paths under each pivot for that repo (not just what the retriever
    happened to return). This makes "list the modules in X" answers
    actually complete instead of being a 5-of-23 sample.
    """
    chunks_list = list(chunks)
    if vector_store is None or not target_repo or not hasattr(vector_store, "enumerate_paths"):
        return build_path_tree_summary(chunks_list, target_repo=target_repo)
    # Probe which pivot is dominant from the retrieved chunks first.
    pivot_counts: dict[str, int] = defaultdict(int)
    for c in chunks_list:
        if (c.repo or "") != target_repo:
            continue
        m = _find_pivot(c.source_path or "")
        if m is not None:
            pivot_counts[m[0]] += 1
    # Query-keyword hint takes precedence -- if the user asked about
    # "charts", we want `charts/` even if the retriever happened to
    # surface chunks dominated by `envs/`. Verify the hint actually has
    # content in the target repo before committing to it.
    hint = _pivot_hint_from_query(query or "")
    pivot: str | None = None
    if hint:
        hinted_paths = await vector_store.enumerate_paths(
            repo=target_repo, path_prefix=hint, max_paths=200,
        )
        if len(hinted_paths) > _MIN_CHUNKS_FOR_SUMMARY:
            pivot = hint
            # Descend one more level if one top-level subdir has WAY
            # more distinct top-2 children than the others. Example:
            # `charts/cluster/<chart>/...` has 61 distinct charts vs
            # `charts/generic-charts/...` which has only a few. The
            # user asking "list all charts" wants the 61 chart names,
            # not the structural buckets `bootstrap/cluster/generic-
            # charts`. We count distinct depth-2 children per top so
            # `generic-charts` (deep but narrow) doesn't outweigh
            # `cluster` (shallow but wide).
            top_depth2_children: dict[str, set[str]] = defaultdict(set)
            for p in hinted_paths:
                after = p[p.find(hint) + len(hint):]
                segs = [s for s in after.split("/") if s]
                if len(segs) >= 2:
                    top_depth2_children[segs[0]].add(segs[1])
            if top_depth2_children:
                dom_top, dom_set = max(
                    top_depth2_children.items(), key=lambda t: len(t[1])
                )
                runner_up = sorted(
                    (len(s) for k, s in top_depth2_children.items() if k != dom_top),
                    reverse=True,
                )
                runner_count = runner_up[0] if runner_up else 0
                # Descend only if the dominant top has at least 10
                # children AND at least 3x as many as the runner-up.
                if len(dom_set) >= 10 and len(dom_set) >= 3 * (runner_count + 1):
                    pivot = hint + dom_top + "/"
    if pivot is None and pivot_counts:
        pivot = max(pivot_counts.items(), key=lambda t: t[1])[0]
    if pivot is None:
        # Fallback: no retrieved chunks landed in the target repo (the
        # retriever returned overview docs from OTHER repos). Try the
        # canonical pivot dirs in priority order against the index;
        # first one with > _MIN_CHUNKS_FOR_SUMMARY distinct paths wins.
        for cand in _PIVOT_DIRS:
            paths = await vector_store.enumerate_paths(
                repo=target_repo, path_prefix=cand, max_paths=_MIN_CHUNKS_FOR_SUMMARY + 1,
            )
            if len(paths) > _MIN_CHUNKS_FOR_SUMMARY:
                pivot = cand
                break
        if pivot is None:
            return build_path_tree_summary(chunks_list, target_repo=target_repo)
    # Enumerate ALL paths under the dominant pivot for the target repo.
    all_paths = await vector_store.enumerate_paths(
        repo=target_repo, path_prefix=pivot, max_paths=5000,
    )
    return _render_tree_for_pivot(
        all_paths=all_paths, pivot=pivot, target_repo=target_repo,
    )


def _render_tree_for_pivot(*, all_paths: list[str], pivot: str, target_repo: str) -> str:
    """Render a 2-level directory tree given a list of paths that ALL
    lie under `pivot` (or include `pivot` somewhere in their path).
    Treats the pivot as authoritative -- does NOT re-detect pivot from
    _PIVOT_DIRS. This is what makes `pivot='charts/cluster/'` work:
    we strip everything up to and including the pivot, then take the
    next 1-2 segments as top/sub.
    """
    tops: dict[str, set[str]] = defaultdict(set)
    chunk_count = 0
    for p in all_paths:
        idx = p.find(pivot)
        if idx < 0:
            continue
        after = p[idx + len(pivot):]
        segs = [s for s in after.split("/") if s]
        if not segs:
            continue
        chunk_count += 1
        top = segs[0]
        # Sub = next segment IFF it's a directory name (no dot -- files
        # like `values.yaml` shouldn't be treated as subdirs).
        sub = segs[1] if len(segs) >= 2 and "." not in segs[1] else ""
        if sub:
            tops[top].add(sub)
        else:
            tops.setdefault(top, set())
    if (
        chunk_count < _MIN_CHUNKS_FOR_SUMMARY
        or len(tops) < _MIN_DISTINCT_TOPS_FOR_SUMMARY
    ):
        return ""
    lines: list[str] = []
    lines.append(
        f"Directory tree under `{pivot}` in `{target_repo}` "
        f"({len(tops)} top-level subdirectories observed):"
    )
    for top in sorted(tops):
        subs = sorted(tops[top])
        if not subs:
            lines.append(f"- `{top}/`")
        elif len(subs) <= _MAX_SUBDIRS_PER_TOP:
            lines.append(f"- `{top}/`  (subdirs: {', '.join(subs)})")
        else:
            shown = subs[:_MAX_SUBDIRS_PER_TOP]
            lines.append(
                f"- `{top}/`  (subdirs: {', '.join(shown)}, ... +{len(subs) - len(shown)} more)"
            )
    lines.append(
        f"Note: this is the COMPLETE tree as indexed for `{target_repo}` "
        f"under `{pivot}` (n={chunk_count} distinct paths enumerated)."
    )
    return "\n".join(lines)


def build_path_tree_summary(
    chunks: Iterable[Chunk],
    *,
    target_repo: str | None = None,
) -> str:
    """Return a markdown summary of the parent/sub-module tree across the
    retrieved chunks, or empty string if no useful structure is present.

    If `target_repo` is given, only chunks whose `repo` field matches it
    are considered. This is the common case: the user named a specific
    repo, the agent should enumerate THAT repo's directory layout, not
    a mixed tree across every retrieved repo.
    """
    chunks_list = list(chunks)
    if not chunks_list:
        return ""

    if target_repo:
        chunks_list = [c for c in chunks_list if (c.repo or "") == target_repo]
        if not chunks_list:
            return ""

    # pivot -> {top: set(subs)}
    by_pivot: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    pivot_counts: dict[str, int] = defaultdict(int)

    for c in chunks_list:
        m = _find_pivot(c.source_path or "")
        if m is None:
            continue
        pivot, rest = m
        top = rest[0]
        sub = rest[1] if len(rest) >= 2 and "." not in rest[1] else ""
        if sub:
            by_pivot[pivot][top].add(sub)
        else:
            # Top-level file directly under pivot -- record the top with no subs.
            by_pivot[pivot].setdefault(top, set())
        pivot_counts[pivot] += 1

    # Pick the dominant pivot -- the one with the most retrieved chunks AND
    # >= _MIN_DISTINCT_TOPS_FOR_SUMMARY distinct tops.
    candidates = [
        (pivot_counts[p], len(by_pivot[p]), p)
        for p in by_pivot
        if pivot_counts[p] >= _MIN_CHUNKS_FOR_SUMMARY
        and len(by_pivot[p]) >= _MIN_DISTINCT_TOPS_FOR_SUMMARY
    ]
    if not candidates:
        return ""
    # Sort by chunk count desc, then by distinct-tops desc.
    candidates.sort(key=lambda t: (-t[0], -t[1]))
    chosen_pivot = candidates[0][2]
    tops = by_pivot[chosen_pivot]

    lines: list[str] = []
    repo_hint = f" in `{target_repo}`" if target_repo else ""
    lines.append(
        f"Directory tree under `{chosen_pivot}`{repo_hint} derived from "
        f"{pivot_counts[chosen_pivot]} retrieved sources "
        f"({len(tops)} top-level subdirectories observed):"
    )
    for top in sorted(tops):
        subs = sorted(tops[top])
        if not subs:
            lines.append(f"- `{top}/`")
        elif len(subs) <= _MAX_SUBDIRS_PER_TOP:
            lines.append(f"- `{top}/`  (subdirs: {', '.join(subs)})")
        else:
            shown = subs[:_MAX_SUBDIRS_PER_TOP]
            lines.append(
                f"- `{top}/`  (subdirs: {', '.join(shown)}, ... +{len(subs) - len(shown)} more)"
            )

    lines.append(
        "Note: this is the *observed* tree from retrieval -- it MAY be "
        "incomplete (a subdir present in the repo but unretrieved won't "
        "appear). If the user asked for the complete list and you're "
        "unsure, say the list is from retrieval and offer to verify "
        "with a repository-tree tool."
    )
    return "\n".join(lines)


def detect_target_repo(
    anchors: list[str],
    chunks: Iterable[Chunk],
) -> str | None:
    """If any anchor token appears in at least one chunk's `repo` value,
    return the repo with the most anchor-matching hits. Used as the
    focus for the directory-tree summary so we enumerate the asked-
    about repo, not a mixed tree.

    Earlier version required EXACTLY one distinct matching repo -- too
    strict because `knowledge_search` legitimately returns chunks from
    Confluence pages whose `repo` is e.g. `confluence:SRE` that ALSO
    mention the target repo's URL. The "winner" pick is fine here
    because we only enumerate paths for the chosen repo; the others
    remain as retrieved-chunk context.
    """
    if not anchors:
        return None
    counts: dict[str, int] = defaultdict(int)
    for c in chunks:
        r = (c.repo or "").lower()
        if not r:
            continue
        for a in anchors:
            if a.lower() in r:
                counts[c.repo] += 1
                break
    if not counts:
        return None
    # Pick the most-hit repo; tie-break by alphabetical order of repo
    # name for determinism.
    return max(counts.items(), key=lambda t: (t[1], t[0]))[0]
