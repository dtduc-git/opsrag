"""GitHub MCP-style tools for OpsRAG (read-only).

Read-only async tools over the GitHub REST API (v3 / 2022-11-28). Works
against github.com and GitHub Enterprise Server (set ``GITHUB_API_URL``
to ``https://<host>/api/v3``).

## Auth

``GITHUB_TOKEN`` (required) -- a fine-grained or classic PAT with read
scopes (``repo`` / ``contents:read`` / ``actions:read`` etc.). Sent as
``Authorization: Bearer <token>``. ``GITHUB_API_URL`` is optional and
defaults to ``https://api.github.com``.

## Read-only enforcement

Every handler issues HTTP GET against the GitHub REST API. The two
"search" tools hit GitHub's ``/search/*`` GET endpoints. No
POST/PUT/DELETE/PATCH anywhere -- nothing mutates a repo, run, PR or
issue.

## Tool list (13 read-only)

| Tool | Endpoint |
|---|---|
| `github_get_file_contents`   | GET `/repos/{o}/{r}/contents/{path}` |
| `github_get_repository_tree` | GET `/repos/{o}/{r}/git/trees/{sha}?recursive=1` |
| `github_search_code`         | GET `/search/code?q=` |
| `github_list_commits`        | GET `/repos/{o}/{r}/commits` |
| `github_get_commit`          | GET `/repos/{o}/{r}/commits/{sha}` |
| `github_list_pull_requests`  | GET `/repos/{o}/{r}/pulls` |
| `github_get_pull_request`    | GET `/repos/{o}/{r}/pulls/{n}` (+ /files) |
| `github_list_issues`         | GET `/repos/{o}/{r}/issues` |
| `github_search_issues`       | GET `/search/issues?q=` |
| `github_list_workflow_runs`  | GET `/repos/{o}/{r}/actions/runs` |
| `github_get_workflow_run`    | GET `/repos/{o}/{r}/actions/runs/{id}` (+ /jobs) |
| `github_get_job_logs`        | GET `/repos/{o}/{r}/actions/jobs/{id}/logs` |
| `github_list_releases`       | GET `/repos/{o}/{r}/releases` |
"""
from __future__ import annotations

import base64
import logging
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from opsrag.mcp.gitlab import MCPTool

_log = logging.getLogger("opsrag.mcp.github")

DEFAULT_API_URL = "https://api.github.com"
GITHUB_API_VERSION = "2022-11-28"
_DEFAULT_TIMEOUT_S = 30.0
_DEFAULT_LIMIT = 25
_MAX_LIMIT = 100
_FILE_TRUNCATE_CHARS = 64000
_LOG_TRUNCATE_CHARS = 32000

# Token env keys, in priority order. Strip shell-export quoting.
_TOKEN_ENV_KEYS = ("GITHUB_TOKEN", "GH_TOKEN")

# Same redaction patterns used by the other log-bearing integrations --
# CI job logs can leak tokens.
_REDACT_PATTERNS = [
    (re.compile(r"\bxoxb-[A-Za-z0-9-]{20,}"), "[REDACTED:slack_bot_token]"),
    (re.compile(r"\bghp_[A-Za-z0-9]{30,}"), "[REDACTED:github_token]"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}"), "[REDACTED:github_pat]"),
    (re.compile(r"\bglpat-[A-Za-z0-9_]{20,}"), "[REDACTED:gitlab_token]"),
    (re.compile(r"\bAKIA[A-Z0-9]{16}\b"), "[REDACTED:aws_access_key]"),
    (re.compile(r"\bAIza[A-Za-z0-9_-]{35}"), "[REDACTED:google_api_key]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]+"), "[REDACTED:jwt]"),
]


def _redact(text: str) -> str:
    for pat, repl in _REDACT_PATTERNS:
        text = pat.sub(repl, text)
    return text


class GitHubMCPError(Exception):
    """Raised on GitHub API errors. Wraps upstream status + body, with
    any token redacted from the message text."""

    def __init__(self, status: int, body: str, *, tool: str | None = None):
        self.status = status
        self.body = _redact(body or "")
        self.tool = tool
        super().__init__(f"[{tool or 'github'}] {status}: {self.body[:300]}")


@dataclass
class _Config:
    token: str
    api_url: str


def _resolve_token() -> str:
    for key in _TOKEN_ENV_KEYS:
        v = os.environ.get(key)
        if v and v.strip():
            return v.strip().strip('"').strip("'")
    return ""


def _config() -> _Config:
    token = _resolve_token()
    if not token:
        raise GitHubMCPError(
            0,
            "GitHub token not set. Set GITHUB_TOKEN (a PAT with read "
            "scopes: repo / contents:read / actions:read).",
            tool="github",
        )
    api_url = (os.environ.get("GITHUB_API_URL") or DEFAULT_API_URL).strip().rstrip("/")
    return _Config(token=token, api_url=api_url)


def _headers() -> dict:
    cfg = _config()
    return {
        "Authorization": f"Bearer {cfg.token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }


async def _get(
    path: str,
    params: dict | None = None,
    *,
    tool: str = "github",
    as_text: bool = False,
) -> Any:
    """Module-level GET. Builds an httpx client from env via _config().
    Follows redirects (job-log download is a 302 to a signed archive URL)."""
    cfg = _config()
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    async with httpx.AsyncClient(
        headers=_headers(), timeout=_DEFAULT_TIMEOUT_S, follow_redirects=True
    ) as http:
        resp = await http.get(f"{cfg.api_url}{path}", params=clean)
    if resp.status_code >= 400:
        raise GitHubMCPError(resp.status_code, resp.text, tool=tool)
    if as_text:
        return resp.text
    return resp.json() if resp.text else {}


def _clamp(n: int | None, default: int = _DEFAULT_LIMIT, *, max: int = _MAX_LIMIT) -> int:
    if n is None:
        return default
    val = int(n)
    if val < 1:
        return 1
    if val > max:
        return max
    return val


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    text = _redact(text)
    if len(text) > limit:
        return text[:limit] + " ...[truncated]"
    return text


def _enc(seg: str | int) -> str:
    """URL-encode one path segment (owner/repo/sha), keeping it a single
    segment. Paths with slashes (file paths, tree shas like 'HEAD') use a
    looser encoding handled by the caller."""
    return quote(str(seg), safe="")


def _user_login(u: Any) -> Any:
    if isinstance(u, dict):
        return u.get("login")
    return u


def _trim_commit(c: dict) -> dict:
    commit = c.get("commit") or {}
    author = commit.get("author") or {}
    return {
        "sha": c.get("sha"),
        "message": _truncate(commit.get("message") or "", 2000),
        "author": author.get("name"),
        "author_email": author.get("email"),
        "date": author.get("date"),
        "login": _user_login(c.get("author")),
        "html_url": c.get("html_url"),
    }


# --- handlers -------------------------------------------------------


async def _h_get_file_contents(_unused, args: dict) -> Any:
    """GET /repos/{owner}/{repo}/contents/{path}?ref=. Decodes base64 file
    content; for a directory, returns the entry list."""
    owner = _enc(args["owner"])
    repo = _enc(args["repo"])
    # Path may legitimately contain slashes; encode each segment.
    raw_path = (args.get("path") or "").strip("/")
    enc_path = "/".join(_enc(p) for p in raw_path.split("/")) if raw_path else ""
    params = {"ref": args.get("ref")}
    resp = await _get(
        f"/repos/{owner}/{repo}/contents/{enc_path}",
        params=params,
        tool="github_get_file_contents",
    )
    if isinstance(resp, list):
        entries = [
            {
                "name": e.get("name"),
                "path": e.get("path"),
                "type": e.get("type"),
                "size": e.get("size"),
                "sha": e.get("sha"),
            }
            for e in resp[: _clamp(args.get("limit"), default=_MAX_LIMIT)]
        ]
        return {
            "owner": args["owner"],
            "repo": args["repo"],
            "path": raw_path,
            "type": "dir",
            "count": len(entries),
            "entries": entries,
        }
    # File object.
    content = resp.get("content") or ""
    encoding = resp.get("encoding")
    decoded: str | None = None
    if encoding == "base64" and content:
        try:
            decoded = base64.b64decode(content).decode("utf-8", errors="replace")
        except (ValueError, UnicodeDecodeError):
            decoded = None
    return {
        "owner": args["owner"],
        "repo": args["repo"],
        "path": resp.get("path"),
        "type": resp.get("type") or "file",
        "size": resp.get("size"),
        "sha": resp.get("sha"),
        "encoding": encoding,
        "html_url": resp.get("html_url"),
        "content": _truncate(decoded, _FILE_TRUNCATE_CHARS) if decoded is not None else None,
        "is_binary": decoded is None and bool(content),
    }


async def _h_get_repository_tree(_unused, args: dict) -> Any:
    """GET /repos/{owner}/{repo}/git/trees/{tree_sha}?recursive=1. tree_sha
    defaults to 'HEAD' (the default branch)."""
    owner = _enc(args["owner"])
    repo = _enc(args["repo"])
    tree_sha = args.get("tree_sha") or "HEAD"
    params = {"recursive": "1" if args.get("recursive", True) else None}
    resp = await _get(
        f"/repos/{owner}/{repo}/git/trees/{_enc(tree_sha)}",
        params=params,
        tool="github_get_repository_tree",
    )
    items = resp.get("tree") or []
    cap = _clamp(args.get("limit"), default=_MAX_LIMIT, max=10000)
    entries = [
        {
            "path": t.get("path"),
            "type": t.get("type"),  # "blob" | "tree"
            "size": t.get("size"),
            "sha": t.get("sha"),
        }
        for t in items[:cap]
    ]
    return {
        "owner": args["owner"],
        "repo": args["repo"],
        "tree_sha": resp.get("sha") or tree_sha,
        "truncated": bool(resp.get("truncated")) or len(items) > cap,
        "count": len(entries),
        "tree": entries,
    }


async def _h_search_code(_unused, args: dict) -> Any:
    """GET /search/code?q=. `q` is required (GitHub code-search syntax,
    e.g. `addClass repo:jquery/jquery`)."""
    q = args["q"]
    params = {
        "q": q,
        "per_page": _clamp(args.get("limit")),
        "page": args.get("page") or 1,
        "sort": args.get("sort"),
        "order": args.get("order"),
    }
    resp = await _get("/search/code", params=params, tool="github_search_code")
    items = resp.get("items") or []
    out = [
        {
            "name": x.get("name"),
            "path": x.get("path"),
            "sha": x.get("sha"),
            "repository": (x.get("repository") or {}).get("full_name"),
            "html_url": x.get("html_url"),
        }
        for x in items
    ]
    return {
        "q": q,
        "total_count": resp.get("total_count"),
        "incomplete_results": resp.get("incomplete_results"),
        "count": len(out),
        "items": out,
    }


async def _h_list_commits(_unused, args: dict) -> Any:
    owner = _enc(args["owner"])
    repo = _enc(args["repo"])
    params = {
        "sha": args.get("sha"),
        "path": args.get("path"),
        "author": args.get("author"),
        "since": args.get("since"),
        "until": args.get("until"),
        "per_page": _clamp(args.get("limit")),
        "page": args.get("page") or 1,
    }
    resp = await _get(
        f"/repos/{owner}/{repo}/commits", params=params, tool="github_list_commits"
    )
    items = resp if isinstance(resp, list) else []
    out = [_trim_commit(c) for c in items]
    return {"owner": args["owner"], "repo": args["repo"], "count": len(out), "commits": out}


async def _h_get_commit(_unused, args: dict) -> Any:
    owner = _enc(args["owner"])
    repo = _enc(args["repo"])
    sha = _enc(args["sha"])
    resp = await _get(
        f"/repos/{owner}/{repo}/commits/{sha}", tool="github_get_commit"
    )
    base = _trim_commit(resp)
    stats = resp.get("stats") or {}
    files = resp.get("files") or []
    base.update(
        {
            "stats": {
                "total": stats.get("total"),
                "additions": stats.get("additions"),
                "deletions": stats.get("deletions"),
            },
            "files": [
                {
                    "filename": f.get("filename"),
                    "status": f.get("status"),
                    "additions": f.get("additions"),
                    "deletions": f.get("deletions"),
                    "changes": f.get("changes"),
                }
                for f in files[:_MAX_LIMIT]
            ],
        }
    )
    return base


async def _h_list_pull_requests(_unused, args: dict) -> Any:
    owner = _enc(args["owner"])
    repo = _enc(args["repo"])
    params = {
        "state": args.get("state") or "open",
        "head": args.get("head"),
        "base": args.get("base"),
        "sort": args.get("sort"),
        "direction": args.get("direction"),
        "per_page": _clamp(args.get("limit")),
        "page": args.get("page") or 1,
    }
    resp = await _get(
        f"/repos/{owner}/{repo}/pulls", params=params, tool="github_list_pull_requests"
    )
    items = resp if isinstance(resp, list) else []
    out = [
        {
            "number": p.get("number"),
            "title": _truncate(p.get("title") or "", 500),
            "state": p.get("state"),
            "draft": p.get("draft"),
            "user": _user_login(p.get("user")),
            "head": (p.get("head") or {}).get("ref"),
            "base": (p.get("base") or {}).get("ref"),
            "created_at": p.get("created_at"),
            "merged_at": p.get("merged_at"),
            "html_url": p.get("html_url"),
        }
        for p in items
    ]
    return {"owner": args["owner"], "repo": args["repo"], "count": len(out), "pull_requests": out}


async def _h_get_pull_request(_unused, args: dict) -> Any:
    """GET /repos/{owner}/{repo}/pulls/{number} plus the changed-files list."""
    owner = _enc(args["owner"])
    repo = _enc(args["repo"])
    number = _enc(args["number"])
    pr = await _get(
        f"/repos/{owner}/{repo}/pulls/{number}", tool="github_get_pull_request"
    )
    files_resp = await _get(
        f"/repos/{owner}/{repo}/pulls/{number}/files",
        params={"per_page": _MAX_LIMIT},
        tool="github_get_pull_request",
    )
    files = files_resp if isinstance(files_resp, list) else []
    return {
        "number": pr.get("number"),
        "title": _truncate(pr.get("title") or "", 500),
        "state": pr.get("state"),
        "draft": pr.get("draft"),
        "merged": pr.get("merged"),
        "merged_at": pr.get("merged_at"),
        "user": _user_login(pr.get("user")),
        "head": (pr.get("head") or {}).get("ref"),
        "base": (pr.get("base") or {}).get("ref"),
        "body": _truncate(pr.get("body") or "", 4000),
        "additions": pr.get("additions"),
        "deletions": pr.get("deletions"),
        "changed_files": pr.get("changed_files"),
        "html_url": pr.get("html_url"),
        "files": [
            {
                "filename": f.get("filename"),
                "status": f.get("status"),
                "additions": f.get("additions"),
                "deletions": f.get("deletions"),
            }
            for f in files
        ],
    }


async def _h_list_issues(_unused, args: dict) -> Any:
    owner = _enc(args["owner"])
    repo = _enc(args["repo"])
    labels = args.get("labels")
    if isinstance(labels, list):
        labels = ",".join(labels)
    params = {
        "state": args.get("state") or "open",
        "labels": labels,
        "assignee": args.get("assignee"),
        "creator": args.get("creator"),
        "since": args.get("since"),
        "sort": args.get("sort"),
        "direction": args.get("direction"),
        "per_page": _clamp(args.get("limit")),
        "page": args.get("page") or 1,
    }
    resp = await _get(
        f"/repos/{owner}/{repo}/issues", params=params, tool="github_list_issues"
    )
    items = resp if isinstance(resp, list) else []
    out = []
    for i in items:
        # The issues endpoint also returns PRs; flag them so callers can tell.
        out.append(
            {
                "number": i.get("number"),
                "title": _truncate(i.get("title") or "", 500),
                "state": i.get("state"),
                "user": _user_login(i.get("user")),
                "labels": [
                    (lbl.get("name") if isinstance(lbl, dict) else lbl)
                    for lbl in (i.get("labels") or [])
                ][:20],
                "comments": i.get("comments"),
                "created_at": i.get("created_at"),
                "is_pull_request": "pull_request" in i,
                "html_url": i.get("html_url"),
            }
        )
    return {"owner": args["owner"], "repo": args["repo"], "count": len(out), "issues": out}


async def _h_search_issues(_unused, args: dict) -> Any:
    """GET /search/issues?q=. `q` required (e.g.
    `repo:octo/api is:issue is:open label:bug`)."""
    q = args["q"]
    params = {
        "q": q,
        "sort": args.get("sort"),
        "order": args.get("order"),
        "per_page": _clamp(args.get("limit")),
        "page": args.get("page") or 1,
    }
    resp = await _get("/search/issues", params=params, tool="github_search_issues")
    items = resp.get("items") or []
    out = [
        {
            "number": x.get("number"),
            "title": _truncate(x.get("title") or "", 500),
            "state": x.get("state"),
            "user": _user_login(x.get("user")),
            "repository_url": x.get("repository_url"),
            "is_pull_request": "pull_request" in x,
            "comments": x.get("comments"),
            "created_at": x.get("created_at"),
            "html_url": x.get("html_url"),
        }
        for x in items
    ]
    return {
        "q": q,
        "total_count": resp.get("total_count"),
        "incomplete_results": resp.get("incomplete_results"),
        "count": len(out),
        "items": out,
    }


async def _h_list_workflow_runs(_unused, args: dict) -> Any:
    owner = _enc(args["owner"])
    repo = _enc(args["repo"])
    params = {
        "status": args.get("status"),
        "branch": args.get("branch"),
        "event": args.get("event"),
        "actor": args.get("actor"),
        "per_page": _clamp(args.get("limit")),
        "page": args.get("page") or 1,
    }
    resp = await _get(
        f"/repos/{owner}/{repo}/actions/runs",
        params=params,
        tool="github_list_workflow_runs",
    )
    runs = resp.get("workflow_runs") if isinstance(resp, dict) else None
    runs = runs or []
    out = [_trim_run(r) for r in runs]
    return {
        "owner": args["owner"],
        "repo": args["repo"],
        "total_count": resp.get("total_count") if isinstance(resp, dict) else len(out),
        "count": len(out),
        "workflow_runs": out,
    }


def _trim_run(r: dict) -> dict:
    return {
        "id": r.get("id"),
        "name": r.get("name"),
        "display_title": _truncate(r.get("display_title") or "", 300),
        "status": r.get("status"),
        "conclusion": r.get("conclusion"),
        "event": r.get("event"),
        "head_branch": r.get("head_branch"),
        "head_sha": r.get("head_sha"),
        "run_number": r.get("run_number"),
        "run_attempt": r.get("run_attempt"),
        "created_at": r.get("created_at"),
        "html_url": r.get("html_url"),
    }


async def _h_get_workflow_run(_unused, args: dict) -> Any:
    """GET /repos/{owner}/{repo}/actions/runs/{run_id} plus its jobs."""
    owner = _enc(args["owner"])
    repo = _enc(args["repo"])
    run_id = _enc(args["run_id"])
    run = await _get(
        f"/repos/{owner}/{repo}/actions/runs/{run_id}",
        tool="github_get_workflow_run",
    )
    jobs_resp = await _get(
        f"/repos/{owner}/{repo}/actions/runs/{run_id}/jobs",
        params={"per_page": _MAX_LIMIT},
        tool="github_get_workflow_run",
    )
    jobs = (jobs_resp.get("jobs") if isinstance(jobs_resp, dict) else None) or []
    out = _trim_run(run)
    out["jobs"] = [
        {
            "id": j.get("id"),
            "name": j.get("name"),
            "status": j.get("status"),
            "conclusion": j.get("conclusion"),
            "started_at": j.get("started_at"),
            "completed_at": j.get("completed_at"),
            "html_url": j.get("html_url"),
            "steps": [
                {
                    "name": s.get("name"),
                    "status": s.get("status"),
                    "conclusion": s.get("conclusion"),
                    "number": s.get("number"),
                }
                for s in (j.get("steps") or [])[:50]
            ],
        }
        for j in jobs
    ]
    return out


async def _h_get_job_logs(_unused, args: dict) -> Any:
    """GET /repos/{owner}/{repo}/actions/jobs/{job_id}/logs. Returns plain
    text (GitHub 302-redirects to a signed archive; httpx follows it).
    Logs are redacted + truncated."""
    owner = _enc(args["owner"])
    repo = _enc(args["repo"])
    job_id = _enc(args["job_id"])
    text = await _get(
        f"/repos/{owner}/{repo}/actions/jobs/{job_id}/logs",
        tool="github_get_job_logs",
        as_text=True,
    )
    full = text or ""
    return {
        "owner": args["owner"],
        "repo": args["repo"],
        "job_id": args["job_id"],
        "total_chars": len(full),
        "logs": _truncate(full, _LOG_TRUNCATE_CHARS),
    }


async def _h_list_releases(_unused, args: dict) -> Any:
    owner = _enc(args["owner"])
    repo = _enc(args["repo"])
    params = {
        "per_page": _clamp(args.get("limit")),
        "page": args.get("page") or 1,
    }
    resp = await _get(
        f"/repos/{owner}/{repo}/releases", params=params, tool="github_list_releases"
    )
    items = resp if isinstance(resp, list) else []
    out = [
        {
            "id": r.get("id"),
            "tag_name": r.get("tag_name"),
            "name": _truncate(r.get("name") or "", 300),
            "draft": r.get("draft"),
            "prerelease": r.get("prerelease"),
            "author": _user_login(r.get("author")),
            "published_at": r.get("published_at"),
            "created_at": r.get("created_at"),
            "body": _truncate(r.get("body") or "", 4000),
            "html_url": r.get("html_url"),
        }
        for r in items
    ]
    return {"owner": args["owner"], "repo": args["repo"], "count": len(out), "releases": out}


# --- tool registry --------------------------------------------------


GITHUB_TOOLS: list[MCPTool] = [
    MCPTool(
        name="github_get_file_contents",
        description=(
            "Read a file or list a directory in a repo. Returns decoded "
            "file text (base64 decoded) or, for a path that is a directory, "
            "the list of entries. Pass `ref` for a branch/tag/SHA."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "path": {"type": "string", "description": "Repo-relative path. Empty for repo root listing."},
                "ref": {"type": "string", "description": "Branch, tag, or commit SHA. Defaults to default branch."},
                "limit": {"type": "number", "description": "Max dir entries."},
            },
            "required": ["owner", "repo", "path"],
        },
        handler=_h_get_file_contents,
    ),
    MCPTool(
        name="github_get_repository_tree",
        description=(
            "List the git tree (all files) of a repo recursively. "
            "`tree_sha` defaults to HEAD (the default branch). Returns "
            "path/type/size for each blob and tree entry."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "tree_sha": {"type": "string", "description": "Tree SHA, branch, or 'HEAD' (default)."},
                "recursive": {"type": "boolean", "description": "Recurse subtrees. Default true."},
                "limit": {"type": "number", "description": "Max entries (cap 10000)."},
            },
            "required": ["owner", "repo"],
        },
        handler=_h_get_repository_tree,
    ),
    MCPTool(
        name="github_search_code",
        description=(
            "Search code across repos with GitHub code-search syntax "
            "(e.g. `q='TODO repo:octo/api language:python'`). Returns "
            "matching file paths + repo. `q` is required."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "q": {"type": "string", "description": "GitHub code-search query."},
                "limit": {"type": "number"},
                "page": {"type": "number"},
                "sort": {"type": "string"},
                "order": {"type": "string", "enum": ["asc", "desc"]},
            },
            "required": ["q"],
        },
        handler=_h_search_code,
    ),
    MCPTool(
        name="github_list_commits",
        description=(
            "List commits on a repo. Filter by `sha` (branch/tag/sha), "
            "`path` (file history), `author`, `since`/`until` (ISO 8601)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "sha": {"type": "string", "description": "Branch, tag, or SHA to start from."},
                "path": {"type": "string"},
                "author": {"type": "string"},
                "since": {"type": "string", "description": "ISO 8601"},
                "until": {"type": "string", "description": "ISO 8601"},
                "limit": {"type": "number"},
                "page": {"type": "number"},
            },
            "required": ["owner", "repo"],
        },
        handler=_h_list_commits,
    ),
    MCPTool(
        name="github_get_commit",
        description="Get a single commit by SHA, with stats and changed files.",
        input_schema={
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "sha": {"type": "string"},
            },
            "required": ["owner", "repo", "sha"],
        },
        handler=_h_get_commit,
    ),
    MCPTool(
        name="github_list_pull_requests",
        description=(
            "List pull requests for a repo. Filter by `state` "
            "(open/closed/all), `head`, `base`."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "state": {"type": "string", "enum": ["open", "closed", "all"]},
                "head": {"type": "string"},
                "base": {"type": "string"},
                "sort": {"type": "string", "enum": ["created", "updated", "popularity", "long-running"]},
                "direction": {"type": "string", "enum": ["asc", "desc"]},
                "limit": {"type": "number"},
                "page": {"type": "number"},
            },
            "required": ["owner", "repo"],
        },
        handler=_h_list_pull_requests,
    ),
    MCPTool(
        name="github_get_pull_request",
        description="Get one PR by number, including its changed-files list.",
        input_schema={
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "number": {"type": "number"},
            },
            "required": ["owner", "repo", "number"],
        },
        handler=_h_get_pull_request,
    ),
    MCPTool(
        name="github_list_issues",
        description=(
            "List issues for a repo. Filter by `state` (open/closed/all), "
            "`labels`, `assignee`, `creator`, `since`. Note GitHub includes "
            "PRs here -- each row carries `is_pull_request`."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "state": {"type": "string", "enum": ["open", "closed", "all"]},
                "labels": {"oneOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}]},
                "assignee": {"type": "string"},
                "creator": {"type": "string"},
                "since": {"type": "string", "description": "ISO 8601"},
                "sort": {"type": "string", "enum": ["created", "updated", "comments"]},
                "direction": {"type": "string", "enum": ["asc", "desc"]},
                "limit": {"type": "number"},
                "page": {"type": "number"},
            },
            "required": ["owner", "repo"],
        },
        handler=_h_list_issues,
    ),
    MCPTool(
        name="github_search_issues",
        description=(
            "Search issues and PRs across repos with GitHub search syntax "
            "(e.g. `q='repo:octo/api is:issue is:open label:bug'`). "
            "`q` is required."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "q": {"type": "string", "description": "GitHub issue-search query."},
                "sort": {"type": "string"},
                "order": {"type": "string", "enum": ["asc", "desc"]},
                "limit": {"type": "number"},
                "page": {"type": "number"},
            },
            "required": ["q"],
        },
        handler=_h_search_issues,
    ),
    MCPTool(
        name="github_list_workflow_runs",
        description=(
            "List GitHub Actions workflow runs for a repo. Filter by "
            "`status` (queued/in_progress/completed/failure/success/...), "
            "`branch`, `event`, `actor`. Use for 'is CI green', 'recent "
            "failed runs'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "status": {"type": "string"},
                "branch": {"type": "string"},
                "event": {"type": "string"},
                "actor": {"type": "string"},
                "limit": {"type": "number"},
                "page": {"type": "number"},
            },
            "required": ["owner", "repo"],
        },
        handler=_h_list_workflow_runs,
    ),
    MCPTool(
        name="github_get_workflow_run",
        description=(
            "Get one Actions workflow run by `run_id`, including its jobs "
            "and per-step status. Use to drill into why a run failed."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "run_id": {"type": "number"},
            },
            "required": ["owner", "repo", "run_id"],
        },
        handler=_h_get_workflow_run,
    ),
    MCPTool(
        name="github_get_job_logs",
        description=(
            "Download the plain-text logs of one Actions job by `job_id`. "
            "Follows GitHub's redirect to the signed archive; logs are "
            "redacted and truncated. Use after `github_get_workflow_run` "
            "to read a failing job's output."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "job_id": {"type": "number"},
            },
            "required": ["owner", "repo", "job_id"],
        },
        handler=_h_get_job_logs,
    ),
    MCPTool(
        name="github_list_releases",
        description="List releases for a repo (tag, name, prerelease/draft flags, notes).",
        input_schema={
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "limit": {"type": "number"},
                "page": {"type": "number"},
            },
            "required": ["owner", "repo"],
        },
        handler=_h_list_releases,
    ),
]


def get_tool(name: str) -> MCPTool:
    for t in GITHUB_TOOLS:
        if t.name == name:
            return t
    raise KeyError(f"unknown github tool: {name}")


# --- fake backend (FR-012; integration tests) ----------------------
#
# GitHub handlers ignore their first (`_unused`) arg and reach the
# module-level `_get`, which builds an httpx client from env via
# `_config()`. The offline fake swaps that single function for a canned,
# shape-faithful responder -- no network, no GITHUB_TOKEN. `build_fake()`
# returns client=None plus a teardown that restores the real `_get`.


_FAKE_FILE_TEXT = "print('hello world')\n"


async def _fake_get(
    path: str,
    params: dict | None = None,
    *,
    tool: str = "github",
    as_text: bool = False,
) -> Any:
    """Canned stand-in for the module-level GET. Routes by path to a
    response shaped like the real GitHub REST endpoint the handler
    parses."""
    if as_text:
        # Job-log archive text.
        return (
            "2026-06-01T00:00:00Z Setup job\n"
            "2026-06-01T00:00:01Z $ pytest -q\n"
            "2026-06-01T00:00:02Z 1 passed\n"
            "2026-06-01T00:00:03Z Job succeeded\n"
        )

    # --- search endpoints ---
    if path == "/search/code":
        return {
            "total_count": 1,
            "incomplete_results": False,
            "items": [
                {
                    "name": "app.py",
                    "path": "src/app.py",
                    "sha": "filesha1",
                    "repository": {"full_name": "octo/api"},
                    "html_url": "https://github.com/octo/api/blob/main/src/app.py",
                }
            ],
        }
    if path == "/search/issues":
        return {
            "total_count": 1,
            "incomplete_results": False,
            "items": [
                {
                    "number": 7,
                    "title": "Crash on startup",
                    "state": "open",
                    "user": {"login": "octocat"},
                    "repository_url": "https://api.github.com/repos/octo/api",
                    "comments": 3,
                    "created_at": "2026-05-01T00:00:00Z",
                    "html_url": "https://github.com/octo/api/issues/7",
                }
            ],
        }

    # --- contents (file vs dir) ---
    if "/contents/" in path or path.endswith("/contents"):
        if path.endswith("/contents/src"):
            return [
                {"name": "app.py", "path": "src/app.py", "type": "file", "size": 21, "sha": "filesha1"},
                {"name": "util.py", "path": "src/util.py", "type": "file", "size": 10, "sha": "filesha2"},
            ]
        return {
            "name": "app.py",
            "path": "src/app.py",
            "type": "file",
            "size": len(_FAKE_FILE_TEXT),
            "sha": "filesha1",
            "encoding": "base64",
            "content": base64.b64encode(_FAKE_FILE_TEXT.encode()).decode(),
            "html_url": "https://github.com/octo/api/blob/main/src/app.py",
        }

    # --- git tree ---
    if "/git/trees/" in path:
        return {
            "sha": "treesha",
            "truncated": False,
            "tree": [
                {"path": "README.md", "type": "blob", "size": 42, "sha": "b1"},
                {"path": "src", "type": "tree", "sha": "t1"},
                {"path": "src/app.py", "type": "blob", "size": 21, "sha": "b2"},
            ],
        }

    # --- commits ---
    if "/commits/" in path:  # single commit
        return {
            "sha": "deadbeef",
            "html_url": "https://github.com/octo/api/commit/deadbeef",
            "commit": {
                "message": "Fix the bug",
                "author": {"name": "Octo Cat", "email": "octo@example.com", "date": "2026-05-01T00:00:00Z"},
            },
            "author": {"login": "octocat"},
            "stats": {"total": 3, "additions": 2, "deletions": 1},
            "files": [
                {"filename": "src/app.py", "status": "modified", "additions": 2, "deletions": 1, "changes": 3}
            ],
        }
    if path.endswith("/commits"):
        return [
            {
                "sha": "deadbeef",
                "html_url": "https://github.com/octo/api/commit/deadbeef",
                "commit": {
                    "message": "Fix the bug",
                    "author": {"name": "Octo Cat", "email": "octo@example.com", "date": "2026-05-01T00:00:00Z"},
                },
                "author": {"login": "octocat"},
            }
        ]

    # --- pull requests ---
    if "/pulls/" in path and path.endswith("/files"):
        return [
            {"filename": "src/app.py", "status": "modified", "additions": 5, "deletions": 2},
        ]
    if "/pulls/" in path:  # single PR
        return {
            "number": 42,
            "title": "Add feature",
            "state": "open",
            "draft": False,
            "merged": False,
            "merged_at": None,
            "user": {"login": "octocat"},
            "head": {"ref": "feature"},
            "base": {"ref": "main"},
            "body": "Implements the thing.",
            "additions": 5,
            "deletions": 2,
            "changed_files": 1,
            "html_url": "https://github.com/octo/api/pull/42",
        }
    if path.endswith("/pulls"):
        return [
            {
                "number": 42,
                "title": "Add feature",
                "state": "open",
                "draft": False,
                "user": {"login": "octocat"},
                "head": {"ref": "feature"},
                "base": {"ref": "main"},
                "created_at": "2026-05-01T00:00:00Z",
                "merged_at": None,
                "html_url": "https://github.com/octo/api/pull/42",
            }
        ]

    # --- issues ---
    if path.endswith("/issues"):
        return [
            {
                "number": 7,
                "title": "Crash on startup",
                "state": "open",
                "user": {"login": "octocat"},
                "labels": [{"name": "bug"}],
                "comments": 3,
                "created_at": "2026-05-01T00:00:00Z",
                "html_url": "https://github.com/octo/api/issues/7",
            }
        ]

    # --- actions: runs / jobs / logs ---
    if path.endswith("/actions/runs"):
        return {
            "total_count": 1,
            "workflow_runs": [_fake_run()],
        }
    if "/actions/runs/" in path and path.endswith("/jobs"):
        return {
            "total_count": 1,
            "jobs": [
                {
                    "id": 9001,
                    "name": "build",
                    "status": "completed",
                    "conclusion": "success",
                    "started_at": "2026-05-01T00:00:00Z",
                    "completed_at": "2026-05-01T00:01:00Z",
                    "html_url": "https://github.com/octo/api/runs/9001",
                    "steps": [
                        {"name": "Set up job", "status": "completed", "conclusion": "success", "number": 1},
                        {"name": "pytest", "status": "completed", "conclusion": "success", "number": 2},
                    ],
                }
            ],
        }
    if "/actions/runs/" in path:  # single run
        return _fake_run()

    # --- releases ---
    if path.endswith("/releases"):
        return [
            {
                "id": 555,
                "tag_name": "v1.2.3",
                "name": "v1.2.3",
                "draft": False,
                "prerelease": False,
                "author": {"login": "octocat"},
                "published_at": "2026-05-01T00:00:00Z",
                "created_at": "2026-04-30T00:00:00Z",
                "body": "Release notes here.",
                "html_url": "https://github.com/octo/api/releases/tag/v1.2.3",
            }
        ]

    return {}


def _fake_run() -> dict:
    return {
        "id": 12345,
        "name": "CI",
        "display_title": "Add feature",
        "status": "completed",
        "conclusion": "success",
        "event": "push",
        "head_branch": "main",
        "head_sha": "deadbeef",
        "run_number": 100,
        "run_attempt": 1,
        "created_at": "2026-05-01T00:00:00Z",
        "html_url": "https://github.com/octo/api/actions/runs/12345",
    }


def build_fake():
    """Return a FakeMCP exposing the GitHub tools wired to an offline
    backend. Needs NO GITHUB_TOKEN / network: the module-level `_get` is
    swapped for a canned responder and restored by `teardown`."""
    import opsrag.mcp.github as _mod
    from opsrag.mcp._fake import FakeMCP

    _orig_get = _mod._get
    _mod._get = _fake_get

    def _restore() -> None:
        _mod._get = _orig_get

    return FakeMCP(tools=list(GITHUB_TOOLS), client=None, teardown=_restore)
