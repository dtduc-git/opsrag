"""GitLab MCP-style tools for OpsRAG (Phase 03 Pillar 1).

Python port of the read-only subset of `@zereight/mcp-gitlab` v2.1.9.
Each tool exposes the MCP shape -- `name`, `description`, `input_schema`
(JSON Schema), and an async handler -- so a future swap to the real
MCP protocol (stdio subprocess) is a one-file refactor.

## Read-only enforcement

All tools are HTTP GET against the GitLab REST API. No POST / PUT /
DELETE / PATCH. The token configured in `GITLAB_PERSONAL_ACCESS_TOKEN`
upstream is the MCP token used by Claude Code's
`@zereight/mcp-gitlab` config; reusing it keeps scope identical and
avoids minting a second SRE-team credential during local development.

## Tool list (10 read-only)

| Tool | GitLab endpoint |
|---|---|
| `gitlab_list_pipelines`         | GET `/projects/:id/pipelines` |
| `gitlab_get_pipeline`           | GET `/projects/:id/pipelines/:pid` |
| `gitlab_list_pipeline_jobs`     | GET `/projects/:id/pipelines/:pid/jobs` |
| `gitlab_get_pipeline_job`       | GET `/projects/:id/jobs/:job_id` + trace tail |
| `gitlab_list_commits`           | GET `/projects/:id/repository/commits` |
| `gitlab_get_commit`             | GET `/projects/:id/repository/commits/:sha` |
| `gitlab_list_merge_requests`    | GET `/merge_requests` or `/projects/:id/merge_requests` |
| `gitlab_get_merge_request`      | GET `/projects/:id/merge_requests/:iid` |
| `gitlab_get_project`            | GET `/projects/:id` |
| `gitlab_list_deployments`       | GET `/projects/:id/deployments` |
"""
from __future__ import annotations

import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

# Strip ANSI/VT100 escape sequences from CI trace logs (CSI / OSC / single-char).
# Pulumi, Terraform, kaniko etc. emit color codes like \x1b[32;1m...\x1b[0m
# which break regex matches on plain text. Pattern covers ESC + non-printable
# terminators commonly seen in CI logs.
_ANSI_RE = re.compile(r"\x1b(?:\[[0-9;?]*[A-Za-z]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])")

# Redact secrets from error bodies before they reach the LLM. CI job
# traces (surfaced via `_h_grep_job_trace` / `gitlab_get_pipeline_job`'s
# `exc.body`) routinely echo tokens on a failed `git clone` / `docker
# login` / deploy step. Same pattern set as the other log-bearing
# integrations (incl. glpat-). Applied at the MCPError source so every
# `exc.body` call-site is safe automatically.
_REDACT_PATTERNS = [
    (re.compile(r"\bxoxb-[A-Za-z0-9-]{20,}"), "[REDACTED:slack_bot_token]"),
    (re.compile(r"\bxoxp-[A-Za-z0-9-]{20,}"), "[REDACTED:slack_user_token]"),
    (re.compile(r"\bghp_[A-Za-z0-9]{30,}"), "[REDACTED:github_token]"),
    (re.compile(r"\bglpat-[A-Za-z0-9_]{20,}"), "[REDACTED:gitlab_token]"),
    (re.compile(r"\bAKIA[A-Z0-9]{16}\b"), "[REDACTED:aws_access_key]"),
    (re.compile(r"\bAIza[A-Za-z0-9_-]{35}"), "[REDACTED:google_api_key]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]+"), "[REDACTED:jwt]"),
    (re.compile(r"\brootly_[A-Za-z0-9_]{30,}"), "[REDACTED:rootly_token]"),
    (re.compile(r"\bddapp_[A-Za-z0-9_]{30,}"), "[REDACTED:dd_app_key]"),
]


def _redact(text: str) -> str:
    for pat, repl in _REDACT_PATTERNS:
        text = pat.sub(repl, text)
    return text

# Per-line text caps to keep payload sane on pathological logs (Pulumi can
# emit single 100KB+ JSON-error lines that explode the tool-result tokens).
_MATCH_LINE_CAP = 500
_CONTEXT_LINE_CAP = 200

DEFAULT_API_URL = "https://gitlab.example.com/api/v4"
_DEFAULT_TIMEOUT_S = 30.0
_DEFAULT_PER_PAGE = 20
_MAX_PER_PAGE = 100
_DEFAULT_LOG_TAIL_LINES = 1000

# Strip surrounding quotes / whitespace common in shell exports.
_TOKEN_ENV_KEYS = (
    "OPSRAG_GITLAB_TOKEN",
    "GITLAB_PERSONAL_ACCESS_TOKEN",
    "GITLAB_TOKEN",
)


class GitLabMCPError(Exception):
    """Raised on GitLab API errors. Wraps upstream status + body, with
    any secrets redacted from both the stored body and the message text.
    Redacting at the source keeps every `exc.body` call-site (e.g.
    `_h_grep_job_trace`) safe automatically."""

    def __init__(self, status: int, body: str, *, tool: str | None = None):
        self.status = status
        self.body = _redact(body or "")
        self.tool = tool
        super().__init__(f"[{tool or 'gitlab'}] {status}: {self.body[:300]}")


@dataclass(frozen=True)
class MCPTool:
    """MCP-shaped tool definition.

    Mirrors the protocol's `Tool` dataclass: a name (unique within the
    server), a human-readable description, a JSON Schema for inputs,
    and an async handler. The handler signature accepts the parsed
    input dict plus the live `GitLabClient`.
    """

    name: str
    description: str
    input_schema: dict
    handler: Callable[[GitLabClient, dict], Awaitable[Any]]

    async def call(self, client: GitLabClient, args: dict) -> Any:
        return await self.handler(client, args)


class GitLabClient:
    """Async GitLab REST client. Read-only."""

    def __init__(
        self,
        token: str | None = None,
        api_url: str | None = None,
        *,
        timeout: float = _DEFAULT_TIMEOUT_S,
        client: httpx.AsyncClient | None = None,
    ):
        token = token or _resolve_token()
        if not token:
            raise RuntimeError(
                "GitLab token not set. Set one of: "
                + ", ".join(_TOKEN_ENV_KEYS)
            )
        self.token = token
        self.api_url = (
            api_url
            or os.environ.get("OPSRAG_GITLAB_API_URL")
            or DEFAULT_API_URL
        ).rstrip("/")
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(
                headers={"PRIVATE-TOKEN": self.token},
                timeout=timeout,
            )
            self._owns_client = True

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> GitLabClient:
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()

    async def get(
        self,
        path: str,
        *,
        params: dict | None = None,
        tool: str | None = None,
        as_text: bool = False,
    ) -> Any:
        url = f"{self.api_url}{path}"
        clean_params = {k: v for k, v in (params or {}).items() if v is not None}
        resp = await self._client.get(url, params=clean_params)
        if resp.status_code >= 400:
            raise GitLabMCPError(resp.status_code, resp.text, tool=tool)
        if as_text:
            return resp.text
        if not resp.text:
            return None
        return resp.json()


def _resolve_token() -> str | None:
    for key in _TOKEN_ENV_KEYS:
        v = os.environ.get(key)
        if v and v.strip():
            return v.strip().strip('"').strip("'")
    return None


def _enc(project_id: str | int) -> str:
    """URL-encode a project ID/path. `123` stays numeric; `saas/acme-notes-be`
    becomes `saas%2Facme-notes-be`."""
    return quote(str(project_id), safe="")


def _clamp_per_page(value: int | None) -> int:
    if value is None:
        return _DEFAULT_PER_PAGE
    return max(1, min(int(value), _MAX_PER_PAGE))


# --- handlers -------------------------------------------------------


async def _h_list_pipelines(client: GitLabClient, args: dict) -> Any:
    pid = _enc(args["project_id"])
    params = {
        "ref": args.get("ref"),
        "sha": args.get("sha"),
        "scope": args.get("scope"),
        "status": args.get("status"),
        "username": args.get("username"),
        "yaml_errors": args.get("yaml_errors"),
        "order_by": args.get("order_by"),
        "sort": args.get("sort"),
        "updated_after": args.get("updated_after"),
        "updated_before": args.get("updated_before"),
        "per_page": _clamp_per_page(args.get("per_page")),
        "page": args.get("page", 1),
    }
    return await client.get(
        f"/projects/{pid}/pipelines", params=params, tool="gitlab_list_pipelines"
    )


async def _h_get_pipeline(client: GitLabClient, args: dict) -> Any:
    pid = _enc(args["project_id"])
    plid = quote(str(args["pipeline_id"]), safe="")
    return await client.get(
        f"/projects/{pid}/pipelines/{plid}", tool="gitlab_get_pipeline"
    )


async def _h_list_pipeline_jobs(client: GitLabClient, args: dict) -> Any:
    pid = _enc(args["project_id"])
    plid = quote(str(args["pipeline_id"]), safe="")
    params = {
        "scope": args.get("scope"),
        "include_retried": args.get("include_retried"),
        "per_page": _clamp_per_page(args.get("per_page")),
        "page": args.get("page", 1),
    }
    return await client.get(
        f"/projects/{pid}/pipelines/{plid}/jobs",
        params=params,
        tool="gitlab_list_pipeline_jobs",
    )


async def _h_get_pipeline_job(client: GitLabClient, args: dict) -> Any:
    """Returns job metadata + tail of the trace log.

    `limit` = number of lines from end (default 1000); `offset` = skip
    that many lines from the end before slicing. Mirrors the
    @zereight/mcp-gitlab semantics so callers can re-walk older log
    chunks without dumping the whole trace.
    """
    pid = _enc(args["project_id"])
    job_id = quote(str(args["job_id"]), safe="")
    job = await client.get(
        f"/projects/{pid}/jobs/{job_id}", tool="gitlab_get_pipeline_job"
    )
    try:
        trace = await client.get(
            f"/projects/{pid}/jobs/{job_id}/trace",
            tool="gitlab_get_pipeline_job",
            as_text=True,
        )
    except GitLabMCPError as exc:
        # Trace endpoint can 404 if the job hasn't started; surface job + reason.
        return {"job": job, "trace": None, "trace_error": str(exc)}

    lines = (trace or "").splitlines()
    limit = int(args.get("limit") or _DEFAULT_LOG_TAIL_LINES)
    offset = int(args.get("offset") or 0)
    end = max(0, len(lines) - offset)
    start = max(0, end - limit)
    tail = "\n".join(lines[start:end])
    return {
        "job": job,
        "trace_total_lines": len(lines),
        "trace_window": {"start": start, "end": end, "lines": end - start},
        "trace": tail,
    }


def _strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences from a CI trace log."""
    if not text:
        return ""
    return _ANSI_RE.sub("", text)


def _truncate(line: str, cap: int) -> str:
    """Trim trailing whitespace then cap at `cap` chars."""
    line = line.rstrip()
    if len(line) > cap:
        return line[:cap]
    return line


async def _h_grep_job_trace(client: GitLabClient, args: dict) -> Any:
    """Grep a job's trace log for `pattern`, return matches + N lines context.

    Designed for "why did job X fail" questions where the actual error is
    buried in the middle of a 2000+ line Pulumi / Terraform / Helm / kaniko
    trace and `gitlab_get_pipeline_job`'s tail-N window misses it.
    """
    project_id = args["project_id"]
    job_id_raw = str(args["job_id"])
    pattern_str = args["pattern"]

    max_matches = int(args.get("max_matches") or 20)
    max_matches = max(1, min(max_matches, 100))
    context_lines = int(args.get("context_lines", 2) if args.get("context_lines") is not None else 2)
    context_lines = max(0, min(context_lines, 5))

    # Compile guard -- bad regex from the LLM should not crash the agent.
    try:
        regex = re.compile(pattern_str)
    except re.error as exc:
        return {"error": f"invalid regex: {exc}"}

    pid = _enc(project_id)
    job_id = quote(job_id_raw, safe="")

    try:
        raw_trace = await client.get(
            f"/projects/{pid}/jobs/{job_id}/trace",
            tool="gitlab_grep_job_trace",
            as_text=True,
        )
    except GitLabMCPError as exc:
        if exc.status in (401, 403):
            return {
                "error": f"unauthorized fetching trace: {exc.body[:200]}",
                "status_code": exc.status,
            }
        if exc.status == 404:
            return {
                "error": f"job or trace not found: {exc.body[:200]}",
                "status_code": exc.status,
            }
        # Other upstream failure -- surface status + short body for the agent.
        return {
            "error": f"gitlab error: {exc.body[:200]}",
            "status_code": exc.status,
        }

    clean = _strip_ansi(raw_trace or "")
    lines = clean.splitlines()
    total_lines = len(lines)

    matches: list[dict[str, Any]] = []
    truncated = False

    for idx, line in enumerate(lines):
        if regex.search(line):
            if len(matches) >= max_matches:
                truncated = True
                break
            before_start = max(0, idx - context_lines)
            after_end = min(total_lines, idx + 1 + context_lines)
            ctx_before = [
                _truncate(lines[i], _CONTEXT_LINE_CAP)
                for i in range(before_start, idx)
            ]
            ctx_after = [
                _truncate(lines[i], _CONTEXT_LINE_CAP)
                for i in range(idx + 1, after_end)
            ]
            matches.append(
                {
                    "line": idx + 1,  # 1-indexed
                    "text": _truncate(line, _MATCH_LINE_CAP),
                    "context_before": ctx_before,
                    "context_after": ctx_after,
                }
            )

    return {
        "project_id": str(project_id),
        "job_id": job_id_raw,
        "pattern": pattern_str,
        "total_matches": len(matches),
        "truncated": truncated,
        "trace_lines": total_lines,
        "matches": matches,
    }


async def _h_list_commits(client: GitLabClient, args: dict) -> Any:
    pid = _enc(args["project_id"])
    params = {
        "ref_name": args.get("ref_name"),
        "since": args.get("since"),
        "until": args.get("until"),
        "path": args.get("path"),
        "author": args.get("author"),
        "all": args.get("all"),
        "with_stats": args.get("with_stats"),
        "first_parent": args.get("first_parent"),
        "order": args.get("order"),
        "trailers": args.get("trailers"),
        "per_page": _clamp_per_page(args.get("per_page")),
        "page": args.get("page", 1),
    }
    return await client.get(
        f"/projects/{pid}/repository/commits",
        params=params,
        tool="gitlab_list_commits",
    )


async def _h_get_commit(client: GitLabClient, args: dict) -> Any:
    pid = _enc(args["project_id"])
    sha = quote(str(args["sha"]), safe="")
    params = {"stats": args.get("stats")}
    return await client.get(
        f"/projects/{pid}/repository/commits/{sha}",
        params=params,
        tool="gitlab_get_commit",
    )


async def _h_list_merge_requests(client: GitLabClient, args: dict) -> Any:
    """Project MRs when `project_id` is given; otherwise the current
    user's MRs across all projects (uses GET /merge_requests)."""
    if "project_id" in args and args["project_id"] is not None:
        path = f"/projects/{_enc(args['project_id'])}/merge_requests"
    else:
        path = "/merge_requests"
    labels = args.get("labels")
    if isinstance(labels, list):
        labels = ",".join(labels)
    params = {
        "state": args.get("state"),
        "scope": args.get("scope"),
        "source_branch": args.get("source_branch"),
        "target_branch": args.get("target_branch"),
        "labels": labels,
        "milestone": args.get("milestone"),
        "search": args.get("search"),
        "author_id": args.get("author_id"),
        "author_username": args.get("author_username"),
        "assignee_id": args.get("assignee_id"),
        "assignee_username": args.get("assignee_username"),
        "reviewer_id": args.get("reviewer_id"),
        "reviewer_username": args.get("reviewer_username"),
        "wip": args.get("wip"),
        "with_labels_details": args.get("with_labels_details"),
        "created_before": args.get("created_before"),
        "created_after": args.get("created_after"),
        "updated_before": args.get("updated_before"),
        "updated_after": args.get("updated_after"),
        "order_by": args.get("order_by"),
        "sort": args.get("sort"),
        "per_page": _clamp_per_page(args.get("per_page")),
        "page": args.get("page", 1),
    }
    return await client.get(path, params=params, tool="gitlab_list_merge_requests")


async def _h_get_merge_request(client: GitLabClient, args: dict) -> Any:
    """Resolves an MR by IID OR by source branch name (matching @zereight
    semantics). Branch lookup uses list-by-source-branch then picks the
    most recent MR -- handles the common 'I know the branch name, not the
    IID' SRE pattern."""
    pid = _enc(args["project_id"])
    iid = args.get("merge_request_iid")
    src = args.get("source_branch")
    if not iid and not src:
        raise GitLabMCPError(
            400,
            "merge_request_iid or source_branch required",
            tool="gitlab_get_merge_request",
        )
    if iid:
        return await client.get(
            f"/projects/{pid}/merge_requests/{quote(str(iid), safe='')}",
            tool="gitlab_get_merge_request",
        )
    matches = await client.get(
        f"/projects/{pid}/merge_requests",
        params={
            "source_branch": src,
            "state": "all",
            "order_by": "updated_at",
            "sort": "desc",
            "per_page": 1,
        },
        tool="gitlab_get_merge_request",
    )
    if not matches:
        raise GitLabMCPError(
            404,
            f"no merge request found for source_branch={src!r}",
            tool="gitlab_get_merge_request",
        )
    return matches[0]


async def _h_get_project(client: GitLabClient, args: dict) -> Any:
    pid = _enc(args["project_id"])
    return await client.get(f"/projects/{pid}", tool="gitlab_get_project")


async def _h_search_projects(client: GitLabClient, args: dict) -> Any:
    """Search GitLab projects by partial name.

    GitLab's `GET /api/v4/projects?search=<q>` returns projects the
    token has access to whose name contains the query. Use this when
    you have a SHORT repo name like ``acme-notes-be`` and need the full
    ``namespace/path-with-namespace`` like ``saas/acme-notes-be`` before
    calling ``gitlab_list_commits`` / ``gitlab_get_pipeline`` / etc.

    Returns trimmed records (``simple=true`` upstream) -- just enough
    to disambiguate (id, name, path_with_namespace, default_branch,
    web_url, visibility).
    """
    params = {
        "search": args["query"],
        # Also match namespace/group names so e.g. `saas` finds every
        # project in that group. Defaults true; opt-out per call.
        "search_namespaces": args.get("search_namespaces", True),
        # `simple=true` strips heavy fields (statistics, namespace
        # tree, etc.) so the response stays under ~5KB even for 20
        # matches. Caller can fetch full project metadata via
        # `gitlab_get_project` once they pick one.
        "simple": True,
        "per_page": _clamp_per_page(args.get("per_page", 10)),
        "page": args.get("page", 1),
        "order_by": args.get("order_by", "last_activity_at"),
        "sort": args.get("sort", "desc"),
    }
    return await client.get(
        "/projects", params=params, tool="gitlab_search_projects",
    )


async def _h_list_tags(client: GitLabClient, args: dict) -> Any:
    """List repo tags. Use for 'latest tag of repo X' / 'tags matching v3.51.*'.

    GitLab `/projects/:id/repository/tags`. Confirmed working shape
    (curl-verified 2026-05-15 against saas/acme-notes-be).
    """
    pid = _enc(args["project_id"])
    params = {
        "order_by": args.get("order_by", "updated"),  # `updated` | `name` | `version`
        "sort": args.get("sort", "desc"),
        "search": args.get("search"),
        "per_page": _clamp_per_page(args.get("per_page")),
        "page": args.get("page", 1),
    }
    return await client.get(
        f"/projects/{pid}/repository/tags",
        params=params, tool="gitlab_list_tags",
    )


async def _h_list_branches(client: GitLabClient, args: dict) -> Any:
    """List repo branches. Use for 'latest release-v* branch', 'is master
    merged into release-v3.51?', 'which branches exist'.

    GitLab `/projects/:id/repository/branches`. Confirmed working shape
    (curl-verified 2026-05-15). `search` supports a literal substring,
    or `^prefix` / `suffix$` anchors. `regex` enables full regex.
    """
    pid = _enc(args["project_id"])
    params = {
        "search": args.get("search"),
        "regex": args.get("regex"),
        "per_page": _clamp_per_page(args.get("per_page")),
        "page": args.get("page", 1),
    }
    return await client.get(
        f"/projects/{pid}/repository/branches",
        params=params, tool="gitlab_list_branches",
    )


async def _h_list_deployments(client: GitLabClient, args: dict) -> Any:
    pid = _enc(args["project_id"])
    params = {
        "environment": args.get("environment"),
        "ref": args.get("ref"),
        "sha": args.get("sha"),
        "status": args.get("status"),
        "order_by": args.get("order_by"),
        "sort": args.get("sort"),
        "updated_after": args.get("updated_after"),
        "updated_before": args.get("updated_before"),
        "per_page": _clamp_per_page(args.get("per_page")),
        "page": args.get("page", 1),
    }
    return await client.get(
        f"/projects/{pid}/deployments",
        params=params,
        tool="gitlab_list_deployments",
    )


# --- tool registry --------------------------------------------------


GITLAB_TOOLS: list[MCPTool] = [
    MCPTool(
        name="gitlab_list_pipelines",
        description="List CI pipelines for a project, with rich filtering "
        "(ref, sha, status, scope, time window, user). Defaults to most-recent first.",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "Project ID or URL-encoded path (e.g. 'saas/acme-notes-be')"},
                "ref": {"type": "string", "description": "Branch / tag ref"},
                "sha": {"type": "string", "description": "Commit SHA"},
                "scope": {"type": "string", "enum": ["running", "pending", "finished", "branches", "tags"]},
                "status": {"type": "string", "enum": [
                    "created", "waiting_for_resource", "preparing", "pending",
                    "running", "success", "failed", "canceled", "skipped",
                    "manual", "scheduled",
                ]},
                "username": {"type": "string"},
                "yaml_errors": {"type": "boolean"},
                "order_by": {"type": "string", "enum": ["id", "status", "ref", "updated_at", "user_id"]},
                "sort": {"type": "string", "enum": ["asc", "desc"]},
                "updated_after": {"type": "string", "description": "ISO 8601"},
                "updated_before": {"type": "string", "description": "ISO 8601"},
                "per_page": {"type": "number", "description": f"Max {_MAX_PER_PAGE}, default {_DEFAULT_PER_PAGE}"},
                "page": {"type": "number"},
            },
            "required": ["project_id"],
        },
        handler=_h_list_pipelines,
    ),
    MCPTool(
        name="gitlab_get_pipeline",
        description="Get details of a specific pipeline (status, duration, ref, sha, user, jobs summary).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "pipeline_id": {"type": "string"},
            },
            "required": ["project_id", "pipeline_id"],
        },
        handler=_h_get_pipeline,
    ),
    MCPTool(
        name="gitlab_list_pipeline_jobs",
        description="List jobs in a pipeline. Filter by scope (failed / success / running).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "pipeline_id": {"type": "string"},
                "scope": {"type": "string", "enum": [
                    "created", "pending", "running", "failed", "success",
                    "canceled", "skipped", "manual",
                ]},
                "include_retried": {"type": "boolean"},
                "per_page": {"type": "number"},
                "page": {"type": "number"},
            },
            "required": ["project_id", "pipeline_id"],
        },
        handler=_h_list_pipeline_jobs,
    ),
    MCPTool(
        name="gitlab_get_pipeline_job",
        description="Get a job's metadata plus the tail of its trace log. "
        "`limit` = lines from end (default 1000), `offset` = skip from end (for pagination).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "job_id": {"type": "string"},
                "limit": {"type": "number"},
                "offset": {"type": "number"},
            },
            "required": ["project_id", "job_id"],
        },
        handler=_h_get_pipeline_job,
    ),
    MCPTool(
        name="gitlab_grep_job_trace",
        description=(
            "Fetch a GitLab CI job's trace log and return lines matching a "
            "regex pattern, with line numbers and surrounding context. Use "
            "this for 'why did job X fail' / 'what errored in pipeline Y' "
            "questions where the trace is long (Pulumi, Terraform, Helm, "
            "kaniko output) and the actual error is buried in the middle, "
            "not in the truncated tail. Returns matching lines with line "
            "numbers + N lines of context before+after, capped at "
            "max_matches. Use grep-style patterns like "
            "\"error|Error \\d+:|googleapi.*Error|FAILED\" -- start broad, "
            "narrow if too many matches."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": (
                        "Project ID or path (e.g. "
                        "'devops/automation-access/cloud-automation-access'). "
                        "Slashes URL-encoded internally -- don't pre-encode."
                    ),
                },
                "job_id": {
                    "type": "string",
                    "description": "GitLab CI job ID, e.g. '5071340'.",
                },
                "pattern": {
                    "type": "string",
                    "description": (
                        "Python `re` regex (NOT egrep). Use \\d, \\b, | etc. "
                        "Matches line-by-line (no multiline flag). Prefix with "
                        "(?i) for case-insensitive match."
                    ),
                },
                "max_matches": {
                    "type": "number",
                    "description": "Max matches to return (1-100). Default 20.",
                },
                "context_lines": {
                    "type": "number",
                    "description": (
                        "Lines of context before AND after each match (0-5). "
                        "Default 2."
                    ),
                },
            },
            "required": ["project_id", "job_id", "pattern"],
        },
        handler=_h_grep_job_trace,
    ),
    MCPTool(
        name="gitlab_list_commits",
        description="List commits on a project, optionally filtered by branch (ref_name), "
        "time range, file path, or author. Default: most-recent on default branch.",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "ref_name": {"type": "string"},
                "since": {"type": "string", "description": "ISO 8601"},
                "until": {"type": "string", "description": "ISO 8601"},
                "path": {"type": "string"},
                "author": {"type": "string"},
                "all": {"type": "boolean"},
                "with_stats": {"type": "boolean"},
                "first_parent": {"type": "boolean"},
                "order": {"type": "string", "enum": ["default", "topo"]},
                "trailers": {"type": "boolean"},
                "per_page": {"type": "number"},
                "page": {"type": "number"},
            },
            "required": ["project_id"],
        },
        handler=_h_list_commits,
    ),
    MCPTool(
        name="gitlab_get_commit",
        description="Get details of a commit by SHA. Set stats=true for line-change counts.",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "sha": {"type": "string", "description": "Commit SHA, branch, or tag"},
                "stats": {"type": "boolean"},
            },
            "required": ["project_id", "sha"],
        },
        handler=_h_get_commit,
    ),
    MCPTool(
        name="gitlab_list_merge_requests",
        description="List merge requests. Without project_id: caller's MRs across all projects. "
        "With project_id: MRs in that project. Common filters: state (opened/merged), "
        "target_branch, source_branch, time windows.",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "state": {"type": "string", "enum": ["opened", "closed", "locked", "merged", "all"]},
                "scope": {"type": "string", "enum": ["created_by_me", "assigned_to_me", "all"]},
                "source_branch": {"type": "string"},
                "target_branch": {"type": "string"},
                "labels": {"oneOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}]},
                "milestone": {"type": "string"},
                "search": {"type": "string"},
                "author_id": {"type": "string"},
                "author_username": {"type": "string"},
                "assignee_id": {"type": "string"},
                "assignee_username": {"type": "string"},
                "reviewer_id": {"type": "string"},
                "reviewer_username": {"type": "string"},
                "wip": {"type": "string", "enum": ["yes", "no"]},
                "with_labels_details": {"type": "boolean"},
                "created_before": {"type": "string"},
                "created_after": {"type": "string"},
                "updated_before": {"type": "string"},
                "updated_after": {"type": "string"},
                "order_by": {"type": "string", "enum": [
                    "created_at", "updated_at", "priority",
                    "label_priority", "milestone_due", "popularity",
                ]},
                "sort": {"type": "string", "enum": ["asc", "desc"]},
                "per_page": {"type": "number"},
                "page": {"type": "number"},
            },
        },
        handler=_h_list_merge_requests,
    ),
    MCPTool(
        name="gitlab_get_merge_request",
        description="Get an MR by its project-internal IID, or look up the latest MR "
        "from a source_branch name. One of {merge_request_iid, source_branch} is required.",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "merge_request_iid": {"type": "string"},
                "source_branch": {"type": "string"},
            },
            "required": ["project_id"],
        },
        handler=_h_get_merge_request,
    ),
    MCPTool(
        name="gitlab_get_project",
        description="Get a project's metadata (id, name, path, default branch, visibility, web url).",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
            },
            "required": ["project_id"],
        },
        handler=_h_get_project,
    ),
    MCPTool(
        name="gitlab_search_projects",
        description=(
            "Find a GitLab project by partial name. Use this FIRST when "
            "you have a short repo name (e.g. `acme-notes-be`) and need the "
            "full path (e.g. `saas/acme-notes-be`) before calling any other "
            "gitlab_* tool that takes a project_id. Avoids guessing "
            "namespace prefixes."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Partial project name. Matches namespace + project name. Case-insensitive.",
                },
                "search_namespaces": {
                    "type": "boolean",
                    "description": "Also match group/subgroup names. Default true.",
                },
                "per_page": {"type": "number"},
                "page": {"type": "number"},
                "order_by": {
                    "type": "string",
                    "enum": ["id", "name", "path", "created_at", "updated_at", "last_activity_at"],
                },
                "sort": {"type": "string", "enum": ["asc", "desc"]},
            },
            "required": ["query"],
        },
        handler=_h_search_projects,
    ),
    MCPTool(
        name="gitlab_list_tags",
        description=(
            "List a project's git tags. Use this for 'latest tag of repo X', "
            "'tags matching v3.51.*', 'tags newest-first'. Each tag includes "
            "the target SHA + the tagged commit's title, author, date, and "
            "web_url. Default sort: updated desc (latest first). "
            "Prefer this over deriving tag names from pipelines -- a tag with "
            "zero pipeline runs would be invisible to that approach."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "order_by": {"type": "string", "enum": ["updated", "name", "version"]},
                "sort": {"type": "string", "enum": ["asc", "desc"]},
                "search": {
                    "type": "string",
                    "description": "Filter by tag name. Plain substring, or `^prefix` / `suffix$` for anchor matches.",
                },
                "per_page": {"type": "number"},
                "page": {"type": "number"},
            },
            "required": ["project_id"],
        },
        handler=_h_list_tags,
    ),
    MCPTool(
        name="gitlab_list_branches",
        description=(
            "List a project's branches. Use this for 'latest release-v* "
            "branch', 'is X merged into master', 'protected branches', "
            "'all branches matching a regex'. Each branch includes "
            "the head commit's metadata + flags (default, protected, merged). "
            "Prefer this over deriving branch names from pipelines."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "search": {
                    "type": "string",
                    "description": "Filter by branch name. Plain substring, or `^prefix` / `suffix$` anchors.",
                },
                "regex": {
                    "type": "string",
                    "description": "Full regex over branch names. Overrides `search` when set.",
                },
                "per_page": {"type": "number"},
                "page": {"type": "number"},
            },
            "required": ["project_id"],
        },
        handler=_h_list_branches,
    ),
    MCPTool(
        name="gitlab_list_deployments",
        description="List deployments for a project -- useful for 'what shipped right before X'. "
        "Filter by environment (e.g. prod), ref, sha, or status.",
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "environment": {"type": "string"},
                "ref": {"type": "string"},
                "sha": {"type": "string"},
                "status": {"type": "string"},
                "order_by": {"type": "string", "enum": [
                    "id", "iid", "created_at", "updated_at", "ref", "status", "environment",
                ]},
                "sort": {"type": "string", "enum": ["asc", "desc"]},
                "updated_after": {"type": "string"},
                "updated_before": {"type": "string"},
                "per_page": {"type": "number"},
                "page": {"type": "number"},
            },
            "required": ["project_id"],
        },
        handler=_h_list_deployments,
    ),
]


def get_tool(name: str) -> MCPTool:
    for t in GITLAB_TOOLS:
        if t.name == name:
            return t
    raise KeyError(f"unknown gitlab tool: {name}")


# --- fake backend (FR-012; integration tests) ----------------------

class _FakeGitLabClient:
    """Offline stand-in for GitLabClient. Returns canned, shape-faithful
    responses keyed by the REST path, with no network. Mirrors the subset
    of GitLabClient the handlers use (`get`, lifecycle hooks)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def aclose(self) -> None:  # parity with GitLabClient
        return None

    async def __aenter__(self) -> _FakeGitLabClient:
        return self

    async def __aexit__(self, *exc_info) -> None:
        return None

    async def get(
        self,
        path: str,
        *,
        params: dict | None = None,
        tool: str | None = None,
        as_text: bool = False,
    ) -> Any:
        self.calls.append((path, dict(params or {})))
        if as_text:
            # job trace text
            return "Running step\n$ make test\nDONE\nJob succeeded\n"
        # Order matters: most specific path fragments first.
        if "/pipelines/" in path and path.endswith("/jobs"):
            return [{"id": 1, "name": "test", "status": "success", "stage": "test"}]
        if "/pipelines/" in path:
            return {"id": 42, "status": "success", "ref": "main", "sha": "deadbeef"}
        if path.endswith("/pipelines"):
            return [{"id": 42, "status": "success", "ref": "main"}]
        if "/merge_requests/" in path:
            return {"iid": 7, "title": "Fix bug", "state": "merged", "author": {"username": "dev"}}
        if path.endswith("/merge_requests"):
            return [{"iid": 7, "title": "Fix bug", "state": "merged"}]
        if "/repository/commits/" in path:
            return {"id": "deadbeef", "title": "Fix bug", "author_name": "Dev"}
        if path.endswith("/repository/commits"):
            return [{"id": "deadbeef", "title": "Fix bug"}]
        if path.endswith("/repository/branches"):
            return [{"name": "main", "default": True}]
        if path.endswith("/repository/tags"):
            return [{"name": "v1.0.0"}]
        if path.endswith("/deployments"):
            return [{"id": 3, "status": "success", "environment": {"name": "prod"}}]
        if path.startswith("/projects/") and "/jobs/" in path:
            return {"id": 1, "name": "test", "status": "success"}
        if path.startswith("/projects/") and path.count("/") == 2:
            return {"id": 1, "path_with_namespace": "group/project", "default_branch": "main"}
        if path == "/projects" or path.startswith("/projects?") or path.startswith("/search"):
            return [{"id": 1, "path_with_namespace": "group/project"}]
        return []


def build_fake():
    """Return a FakeMCP exposing the GitLab tools bound to an offline client."""
    from opsrag.mcp._fake import FakeMCP

    return FakeMCP(tools=list(GITLAB_TOOLS), client=_FakeGitLabClient())
