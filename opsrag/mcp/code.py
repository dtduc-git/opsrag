"""Agentic code-exploration MCP tools (P1).

Five read-only tools that let the LLM *explore* the codebase at query
time, mirroring Claude Code / Cline / Aider's tool palette:

  - `code_list_repos`  -- enumerate available repos
  - `code_glob`        -- list files matching a path glob
  - `code_grep`        -- git grep regex; returns file:line:match
  - `code_read_file`   -- bounded line-range read
  - `code_find_symbol` -- heuristic search for class/function/route declarations

Source of truth: the shallow-cloned repo cache at `/tmp/opsrag-repos`.
On the indexer pod the daily auto-index loop pre-populates this whole
cache. On the backend pod (which doesn't share a volume with the
indexer) we LAZY-CLONE on first miss using `bind_scm`-supplied
GitCloneSCM. After a backend pod restart the first query touching each
repo pays ~2-30s of clone time; subsequent queries hit the warm
emptyDir cache.

Safety:
  - Path traversal blocked (resolved paths must stay within the cache root).
  - Subprocess timeouts (5s grep, 2s ls).
  - Result caps: at most 200 grep hits, 500 glob files, 500 lines read.
  - Read-only. Never writes, never deletes, never mutates git state.
  - Uses `asyncio.create_subprocess_exec` (no shell expansion of args).
  - Lazy clone is GATED on the configured-repo allowlist -- the agent
    can't trick us into cloning arbitrary repos by passing a malicious
    `repo` argument.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from opsrag.mcp.dependency_parsers import (
    is_dependency_file,
    list_dependencies,
    resolve_dependency,
)
from opsrag.mcp.gitlab import MCPTool

_log = logging.getLogger("opsrag.mcp.code")

_CACHE_ROOT = Path(os.environ.get("OPSRAG_REPO_CACHE", "/tmp/opsrag-repos"))

# -- Lazy-clone state (bound at lifespan startup via `bind_scm`) ---------
# When set, tool handlers will call `_ensure_cloned(repo)` before
# touching the filesystem and that helper shallow-clones the repo on
# demand. When unset, tools fall back to the pre-existing "read what's
# already on disk" behavior (works fine on the indexer pod where the
# auto-index loop pre-clones everything).
_scm: Any = None
_repo_to_branch: dict[str, str] = {}
# Per-repo locks to dedupe concurrent clones of the same repo across
# parallel tool calls. Lazily created in `_ensure_cloned`.
_clone_locks: dict[str, asyncio.Lock] = {}
_clone_locks_guard = asyncio.Lock()


def bind_scm(scm: Any, repos_with_branch: list[tuple[str, str]]) -> None:
    """Inject the SCM provider so `_ensure_cloned` can lazy-clone.

    `repos_with_branch` is the canonical [(repo, branch)] list from
    SCMConfig.repos_with_branch() -- it acts as the allowlist so the
    agent can't ask us to clone arbitrary external repos by passing
    a crafted `repo` argument.
    """
    global _scm, _repo_to_branch
    _scm = scm
    _repo_to_branch = {repo: branch for repo, branch in repos_with_branch}
    _log.info(
        "code MCP tools bound: scm=%s configured_repos=%d",
        type(scm).__name__, len(_repo_to_branch),
    )


async def _ensure_cloned(repo: str) -> tuple[Path | None, str | None]:
    """Return `(repo_dir, None)` if the repo is (or becomes) available on
    disk, or `(None, error_message)` if we can't make it so.

    On a cache hit, returns immediately. On miss:
      1. If SCM isn't bound (legacy path / indexer pod), return None with
         a hint -- caller will respond with "repo not in cache".
      2. If `repo` isn't in the configured allowlist, refuse to clone.
         Protects against the agent fabricating repo names.
      3. Acquire a per-repo lock, double-check, then shallow-clone via
         GitCloneSCM._ensure_cloned. Lock prevents two concurrent
         tool calls from racing to clone the same repo.

    Errors during clone propagate as the second tuple element so the
    caller can include them in the user-facing error payload.
    """
    repo_dir = _CACHE_ROOT / _flatten_repo(repo)
    if repo_dir.is_dir() and (repo_dir / ".git").exists():
        return repo_dir, None

    if _scm is None:
        return None, f"repo {repo} not in cache and SCM not bound for lazy-clone"
    if repo not in _repo_to_branch:
        return None, (
            f"repo {repo} is not in OpsRAG's configured repo allowlist; "
            f"refusing to clone arbitrary repos"
        )

    async with _clone_locks_guard:
        lock = _clone_locks.setdefault(repo, asyncio.Lock())

    async with lock:
        # Double-check post-acquire -- a sibling coroutine may have
        # cloned while we waited on the lock.
        if repo_dir.is_dir() and (repo_dir / ".git").exists():
            return repo_dir, None
        branch = _repo_to_branch[repo]
        try:
            _log.info("code MCP: lazy-cloning %s @ %s", repo, branch)
            t0 = time.time()
            await _scm._ensure_cloned(repo, branch=branch)
            _log.info(
                "code MCP: cloned %s @ %s in %.1fs",
                repo, branch, time.time() - t0,
            )
            return repo_dir, None
        except Exception as exc:  # noqa: BLE001
            _log.warning("code MCP: clone %s failed: %s", repo, exc)
            return None, f"clone failed: {exc}"

_GREP_MAX_HITS = 200
_GLOB_MAX_FILES = 500
_READ_MAX_LINES = 500
_READ_MAX_BYTES = 200_000
_SUBPROCESS_TIMEOUT = 5.0

_REPO_SEP = "__"


def _flatten_repo(repo: str) -> str:
    return repo.strip("/").replace("/", _REPO_SEP)


def _repo_dir(repo: str) -> Path | None:
    if not repo:
        return None
    flat = _flatten_repo(repo)
    candidate = _CACHE_ROOT / flat
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    try:
        resolved.relative_to(_CACHE_ROOT.resolve())
    except ValueError:
        return None
    if not resolved.is_dir():
        return None
    return resolved


def _safe_relative_path(repo_dir: Path, rel_path: str) -> Path | None:
    if not rel_path:
        return None
    candidate = (repo_dir / rel_path.lstrip("/")).resolve()
    try:
        candidate.relative_to(repo_dir.resolve())
    except ValueError:
        return None
    return candidate if candidate.exists() else None


async def _run_git(args: list[str], cwd: Path, timeout: float = _SUBPROCESS_TIMEOUT) -> tuple[int, str, str]:
    """Spawn `git <args>` via asyncio.create_subprocess_exec.

    No shell, no string concatenation -- args are passed as a list to
    the kernel-level execve, eliminating shell-metacharacter injection
    even if `args` came from untrusted input. Timeout enforced; on
    expiry the process is killed and the call returns (124, "", "...").
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", *args,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return 124, "", f"git {args[0]} timed out after {timeout}s"
        return proc.returncode or 0, stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")
    except FileNotFoundError:
        return 127, "", "git binary not found in container"


async def _h_code_list_repos(_unused, _args: dict) -> Any:
    """Return the union of (configured-repo allowlist) U (already-cloned).

    On the indexer pod, on-disk == configured because auto-index pre-clones
    everything. On the backend pod (lazy-clone path), on-disk grows over
    time as queries arrive -- but we still surface the FULL configured list
    so the reasoner can route `code_grep(repo='saas/acme-notes-be', ...)` even
    before that repo has been cloned. The lazy-clone fires inside the
    next tool call.
    """
    on_disk: set[str] = set()
    if _CACHE_ROOT.is_dir():
        for child in sorted(_CACHE_ROOT.iterdir()):
            if not child.is_dir() or not (child / ".git").exists():
                continue
            on_disk.add(child.name.replace(_REPO_SEP, "/"))

    configured = set(_repo_to_branch.keys())
    repos = sorted(on_disk | configured)
    return {
        "repos": repos,
        "count": len(repos),
        "cached": sorted(on_disk),
        "lazy_clone_enabled": _scm is not None,
    }


async def _h_code_glob(_unused, args: dict) -> Any:
    repo = (args.get("repo") or "").strip()
    pattern = (args.get("pattern") or "").strip()
    if not repo or not pattern:
        return {"error": "both `repo` and `pattern` are required"}
    repo_dir, clone_err = await _ensure_cloned(repo)
    if repo_dir is None:
        return {"error": clone_err or f"repo '{repo}' not in cache"}

    code, out, err = await _run_git(["ls-files", "--", pattern], cwd=repo_dir, timeout=4.0)
    if code != 0:
        return {"error": f"git ls-files failed: {err.strip() or 'non-zero exit'}", "repo": repo, "pattern": pattern}

    files = [ln for ln in out.splitlines() if ln]
    truncated = False
    if len(files) > _GLOB_MAX_FILES:
        files = files[:_GLOB_MAX_FILES]
        truncated = True
    return {
        "repo": repo,
        "pattern": pattern,
        "count": len(files),
        "truncated": truncated,
        "files": files,
    }


async def _h_code_grep(_unused, args: dict) -> Any:
    repo = (args.get("repo") or "").strip()
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        return {"error": "`pattern` (regex) is required"}
    if not repo:
        return {"error": "`repo` is required; call `code_list_repos` first if you don't know it"}
    path_glob = (args.get("path_glob") or "").strip()
    case_insensitive = bool(args.get("case_insensitive"))
    max_hits = int(args.get("max_hits") or _GREP_MAX_HITS)
    max_hits = max(1, min(max_hits, _GREP_MAX_HITS))

    repo_dir, clone_err = await _ensure_cloned(repo)
    if repo_dir is None:
        return {"error": clone_err or f"repo '{repo}' not in cache"}

    git_args = ["grep", "-n", "-E", "--full-name"]
    if case_insensitive:
        git_args.append("-i")
    git_args.append(pattern)
    if path_glob:
        git_args += ["--", path_glob]

    code, out, err = await _run_git(git_args, cwd=repo_dir, timeout=_SUBPROCESS_TIMEOUT)
    if code not in (0, 1):
        return {"error": f"git grep failed (exit {code}): {err.strip() or '(no stderr)'}", "repo": repo, "pattern": pattern}

    hits: list[dict] = []
    for line in out.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        path, lineno, text = parts
        try:
            lineno_i = int(lineno)
        except ValueError:
            continue
        hits.append({"path": path, "line": lineno_i, "text": text[:400]})
        if len(hits) >= max_hits:
            break

    return {
        "repo": repo,
        "pattern": pattern,
        "path_glob": path_glob or None,
        "case_insensitive": case_insensitive,
        "count": len(hits),
        "hits": hits,
    }


async def _h_code_read_file(_unused, args: dict) -> Any:
    repo = (args.get("repo") or "").strip()
    path = (args.get("path") or "").strip()
    if not repo or not path:
        return {"error": "both `repo` and `path` are required"}
    repo_dir, clone_err = await _ensure_cloned(repo)
    if repo_dir is None:
        return {"error": clone_err or f"repo '{repo}' not in cache"}
    abs_path = _safe_relative_path(repo_dir, path)
    if abs_path is None or not abs_path.is_file():
        return {"error": f"file '{path}' not found in repo '{repo}'"}

    start = max(1, int(args.get("start_line") or 1))
    end_arg = args.get("end_line")
    end = int(end_arg) if end_arg is not None else (start + _READ_MAX_LINES - 1)
    if end < start:
        return {"error": "end_line must be >= start_line"}
    end = min(end, start + _READ_MAX_LINES - 1)

    try:
        data = abs_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"error": f"read failed: {exc}"}

    if len(data) > _READ_MAX_BYTES * 4:
        data = data[: _READ_MAX_BYTES * 4]

    all_lines = data.splitlines()
    total_lines = len(all_lines)
    end = min(end, total_lines)
    slice_ = all_lines[start - 1: end]

    joined = "\n".join(slice_)
    truncated_by_bytes = False
    if len(joined) > _READ_MAX_BYTES:
        joined = joined[: _READ_MAX_BYTES]
        truncated_by_bytes = True

    return {
        "repo": repo,
        "path": path,
        "start_line": start,
        "end_line": end,
        "total_lines": total_lines,
        "truncated_by_bytes": truncated_by_bytes,
        "content": joined,
    }


_SYMBOL_PATTERNS = {
    "python": [
        r"^[[:space:]]*(?:async[[:space:]]+)?def[[:space:]]+{NAME}\b",
        r"^[[:space:]]*class[[:space:]]+{NAME}\b",
    ],
    "typescript": [
        r"^[[:space:]]*(?:export[[:space:]]+)?(?:async[[:space:]]+)?function[[:space:]]+{NAME}\b",
        r"^[[:space:]]*(?:export[[:space:]]+)?class[[:space:]]+{NAME}\b",
        r"^[[:space:]]*(?:export[[:space:]]+)?const[[:space:]]+{NAME}[[:space:]]*=",
    ],
    "go": [
        r"^func[[:space:]]+(?:\([^)]*\)[[:space:]]+)?{NAME}\b",
        r"^type[[:space:]]+{NAME}\b",
    ],
    "shell": [
        r"^[[:space:]]*{NAME}\(\)",
        r"^[[:space:]]*function[[:space:]]+{NAME}\b",
    ],
}

_KIND_TO_GLOBS = {
    "python": ["*.py"],
    "typescript": ["*.ts", "*.tsx", "*.js", "*.jsx"],
    "go": ["*.go"],
    "shell": ["*.sh", "*.bash"],
}


def _build_symbol_regex(name: str, kind: str | None) -> tuple[str, list[str]]:
    escaped = re.escape(name)
    languages = [kind] if kind in _SYMBOL_PATTERNS else list(_SYMBOL_PATTERNS.keys())
    raw_patterns: list[str] = []
    globs: set[str] = set()
    for lang in languages:
        for p in _SYMBOL_PATTERNS[lang]:
            raw_patterns.append(p.replace("{NAME}", escaped))
        for g in _KIND_TO_GLOBS.get(lang, []):
            globs.add(g)
    combined = "|".join(f"(?:{p})" for p in raw_patterns)
    return combined, sorted(globs)


async def _h_code_find_symbol(_unused, args: dict) -> Any:
    name = (args.get("name") or "").strip()
    if not name:
        return {"error": "`name` is required"}
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        return {"error": "`name` must be a single identifier ([A-Za-z_][A-Za-z0-9_]*)"}
    repo = (args.get("repo") or "").strip()
    kind = (args.get("kind") or "").strip().lower() or None
    if kind and kind not in _SYMBOL_PATTERNS:
        return {"error": f"`kind` must be one of {sorted(_SYMBOL_PATTERNS.keys())} or omitted"}

    regex, globs = _build_symbol_regex(name, kind)

    repos: list[str]
    if repo:
        d, clone_err = await _ensure_cloned(repo)
        if d is None:
            return {"error": clone_err or f"repo '{repo}' not in cache"}
        repos = [repo]
    else:
        # No specific repo -- fan out across whatever is ALREADY on disk.
        # Don't trigger lazy-clone of every configured repo from one call;
        # that'd take minutes. The agent should narrow to a specific repo
        # when it knows which one to look in.
        on_disk: list[str] = []
        if _CACHE_ROOT.is_dir():
            for child in _CACHE_ROOT.iterdir():
                if child.is_dir() and (child / ".git").exists():
                    on_disk.append(child.name.replace(_REPO_SEP, "/"))
        repos = sorted(on_disk)

    all_hits: list[dict] = []
    for r in repos:
        d = _repo_dir(r)
        if d is None:
            continue
        git_args = ["grep", "-n", "-E", "--full-name", regex]
        if globs:
            git_args.append("--")
            git_args.extend(globs)
        code, out, _ = await _run_git(git_args, cwd=d, timeout=3.0)
        if code not in (0, 1):
            continue
        for line in out.splitlines():
            parts = line.split(":", 2)
            if len(parts) != 3:
                continue
            path, lineno, text = parts
            try:
                lineno_i = int(lineno)
            except ValueError:
                continue
            all_hits.append({
                "repo": r,
                "path": path,
                "line": lineno_i,
                "text": text[:400],
            })
            if len(all_hits) >= _GREP_MAX_HITS:
                break
        if len(all_hits) >= _GREP_MAX_HITS:
            break

    return {
        "name": name,
        "kind": kind,
        "repos_searched": len(repos),
        "count": len(all_hits),
        "hits": all_hits,
    }


# Directories never worth scanning for manifests (installed deps, build out,
# vcs). Keeps the dependency-lookup walk bounded on large repos.
_DEP_SKIP_DIRS = frozenset({
    "node_modules", ".venv", "venv", "vendor", "dist", "build", ".git",
    ".cache", "target", ".next", ".nuxt", ".turbo", "__pycache__",
})
_DEP_MAX_FILES = 200
_DEP_MAX_BYTES = 1_000_000


async def _h_code_dependency_lookup(_unused, args: dict) -> Any:
    """Parse a repo's manifests + lockfiles to resolve a dependency's version.

    Unlike `code_grep`, this reads the LOCKFILE (so it returns the resolved,
    pinned version, not just the manifest's range) and matches the package
    name CASE-INSENSITIVELY. Optional `path` scopes the scan to a monorepo
    sub-directory."""
    repo = (args.get("repo") or "").strip()
    package = (args.get("package") or "").strip()
    if not repo:
        return {"error": "`repo` is required"}
    # `package` is optional: omit it to LIST every dependency in the repo
    # (answers "what libraries/versions does X use?"); pass it to resolve one.
    repo_dir, clone_err = await _ensure_cloned(repo)
    if repo_dir is None:
        return {"error": clone_err or f"repo '{repo}' not in cache"}

    sub = (args.get("path") or "").strip()
    if sub:
        search_root = _safe_relative_path(repo_dir, sub)
        if search_root is None or not search_root.is_dir():
            return {"error": f"path '{sub}' not found in repo '{repo}'"}
    else:
        search_root = repo_dir

    files: dict[str, str] = {}
    scanned = 0
    truncated = False
    for dirpath, dirnames, filenames in os.walk(search_root):
        dirnames[:] = [d for d in dirnames if d not in _DEP_SKIP_DIRS]
        for fn in filenames:
            if not is_dependency_file(fn):
                continue
            fp = Path(dirpath) / fn
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")[:_DEP_MAX_BYTES]
                rel = str(fp.relative_to(repo_dir))
            except (OSError, ValueError):
                continue
            files[rel] = text
            scanned += 1
            if scanned >= _DEP_MAX_FILES:
                truncated = True
                break
        if scanned >= _DEP_MAX_FILES:
            break

    if not files:
        base = {
            "repo": repo, "found": False, "findings": [],
            "note": "no dependency manifests/lockfiles found"
                    + (f" under '{sub}'" if sub else " in repo root tree"),
        }
        if package:
            base.update(package=package, resolved_version=None, declared_constraint=None)
        else:
            base.update(count=0, ecosystems=[], dependencies=[])
        return base

    # No package -> LIST every dependency (answers "what libs/versions does X use?").
    if not package:
        result = list_dependencies(files)
        result["repo"] = repo
        result["searched_files"] = sorted(files)
        if truncated:
            result["files_truncated"] = True
        if not result["found"]:
            result["note"] = (
                f"parsed {len(files)} manifest/lockfile(s) but extracted no "
                "dependencies (files may be empty or in an unsupported format)."
            )
        return result

    result = resolve_dependency(files, package)
    result["repo"] = repo
    result["searched_files"] = sorted(files)
    if truncated:
        result["truncated"] = True
    if not result["found"]:
        result["note"] = (
            f"'{package}' not found in {len(files)} manifest/lockfile(s). "
            "It may be a transitive dependency, named differently, or in a "
            "sub-path not scanned (try the `path` argument)."
        )
    elif result["resolved_version"] is None:
        result["note"] = (
            "Only a declared constraint was found (no lockfile entry). The "
            "exact installed version isn't pinned in a committed lockfile."
        )
    return result


_FAKE_REPO = "group/example-repo"
_FAKE_BRANCH = "main"
_FAKE_FILES = {
    "README.md": (
        "# example-repo\n"
        "\n"
        "Sample repository used by the offline code-search fake.\n"
    ),
    "src/app.py": (
        "import logging\n"
        "\n"
        "_log = logging.getLogger(__name__)\n"
        "\n"
        "\n"
        "def handle_request(payload):\n"
        '    """Entry point referenced by the fake tests."""\n'
        "    _log.info(\"handling request\")\n"
        "    return {\"status\": \"ok\"}\n"
        "\n"
        "\n"
        "class RequestHandler:\n"
        "    def run(self):\n"
        "        return handle_request({})\n"
    ),
    "src/util.go": (
        "package util\n"
        "\n"
        "func Add(a int, b int) int {\n"
        "    return a + b\n"
        "}\n"
    ),
    # Dependency manifests + lockfiles for the code_dependency_lookup fake.
    # Manifest declares a range ('^2.3'); lockfile pins the resolved version.
    "pyproject.toml": (
        "[project]\n"
        'name = "example"\n'
        'version = "0.1.0"\n'
        "dependencies = [\n"
        '    "Flask>=2.3",\n'
        '    "requests",\n'
        "]\n"
    ),
    "poetry.lock": (
        "[[package]]\n"
        'name = "flask"\n'
        'version = "2.3.3"\n'
        "\n"
        "[[package]]\n"
        'name = "requests"\n'
        'version = "2.31.0"\n'
    ),
    "web/package.json": (
        "{\n"
        '  "name": "frontend",\n'
        '  "dependencies": { "react": "^18.2.0" }\n'
        "}\n"
    ),
    "web/package-lock.json": (
        "{\n"
        '  "lockfileVersion": 3,\n'
        '  "packages": {\n'
        '    "node_modules/react": { "version": "18.2.0" }\n'
        "  }\n"
        "}\n"
    ),
}


def build_fake():
    """Return a FakeMCP exposing the code tools over an offline temp repo.

    Faking approach (self-contained, no network, no real checkout): create a
    throwaway temporary directory, initialise a real git repo inside it with a
    couple of canned files, commit them, then point the module's `_CACHE_ROOT`
    at that temp directory and bind the repo into the configured allowlist via
    `bind_scm(None, ...)`. Because the repo already exists on disk,
    `_ensure_cloned` returns a cache hit and never attempts a clone (SCM is
    None, so no network is reachable even if it tried). The handlers then run
    their normal `git grep` / `git ls-files` subprocesses against the temp repo,
    so results are real and shape-faithful rather than mocked.

    teardown restores `_CACHE_ROOT`, unbinds the SCM state, and removes the
    temp tree.
    """
    import shutil
    import subprocess
    import tempfile

    from opsrag.mcp._fake import FakeMCP

    global _CACHE_ROOT, _scm, _repo_to_branch, _clone_locks

    prev_cache_root = _CACHE_ROOT
    prev_scm = _scm
    prev_repo_to_branch = _repo_to_branch
    prev_clone_locks = _clone_locks

    tmp_root = Path(tempfile.mkdtemp(prefix="opsrag-code-fake-"))
    repo_dir = tmp_root / _flatten_repo(_FAKE_REPO)
    repo_dir.mkdir(parents=True, exist_ok=True)

    def _git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=str(repo_dir),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    _git("init", "-q", "-b", _FAKE_BRANCH)
    _git("config", "user.email", "fake@example.com")
    _git("config", "user.name", "Fake Tester")
    for rel, content in _FAKE_FILES.items():
        target = repo_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    _git("add", "-A")
    _git("commit", "-q", "-m", "seed fake repo")

    # Point the module at the temp cache and register the repo allowlist so
    # `_ensure_cloned` hits the on-disk repo without ever cloning.
    _CACHE_ROOT = tmp_root
    _scm = None
    _repo_to_branch = {_FAKE_REPO: _FAKE_BRANCH}
    _clone_locks = {}

    def _teardown() -> None:
        global _CACHE_ROOT, _scm, _repo_to_branch, _clone_locks
        _CACHE_ROOT = prev_cache_root
        _scm = prev_scm
        _repo_to_branch = prev_repo_to_branch
        _clone_locks = prev_clone_locks
        shutil.rmtree(tmp_root, ignore_errors=True)

    return FakeMCP(tools=list(CODE_TOOLS), client=None, teardown=_teardown)


CODE_TOOLS: list[MCPTool] = [
    MCPTool(
        name="code_list_repos",
        description=(
            "List every repo currently available in the local "
            "code cache. Call this FIRST when you don't know the exact repo "
            "name for a code-exploration query -- the other `code_*` tools "
            "require a repo argument. Output is the human-readable form "
            "(e.g. `devops/gitops`, `saas/acme-notes-be`)."
        ),
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=_h_code_list_repos,
    ),
    MCPTool(
        name="code_glob",
        description=(
            "List file paths inside a repo that match a path glob. Honors "
            "`.gitignore` (uses `git ls-files` under the hood). Use this to "
            "scope a subsequent `code_grep` to a directory, or to enumerate "
            "files of a given extension before deciding which to read.\n\n"
            "Examples:\n"
            "  - `pattern='values/saas/acme-notes-be/*.yaml'` -- Helm value files for acme-notes-be\n"
            "  - `pattern='**/*.tf'` -- all Terraform files in the repo\n"
            "  - `pattern='src/app/**/*.component.ts'` -- Angular components"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repo name in `group/name` form, e.g. `devops/gitops`."},
                "pattern": {"type": "string", "description": "Git pathspec / glob, e.g. `values/saas/*/prod.yaml`."},
            },
            "required": ["repo", "pattern"],
            "additionalProperties": False,
        },
        handler=_h_code_glob,
    ),
    MCPTool(
        name="code_grep",
        description=(
            "Run `git grep -E` regex inside a repo. Returns up to 200 "
            "`{path, line, text}` hits. Use this as the workhorse for "
            "'where in the code is X' and 'which file defines / references "
            "Y' questions. Always prefer this over `knowledge_search` for "
            "code-shape queries (function names, route paths, identifiers) "
            "-- exact match beats embedding similarity for symbol lookup.\n\n"
            "Examples:\n"
            "  - `pattern='int-widget-legacy', repo='devops/gitops'` -- find the Kong route\n"
            "  - `pattern='class WidgetViewSet', repo='saas/acme-notes-be'` -- find the Django view\n"
            "  - `pattern='acme-analytics-v3', path_glob='values/saas/**'` -- find every reference in gitops values"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repo name. Call `code_list_repos` first if unsure."},
                "pattern": {"type": "string", "description": "Regex (POSIX extended). Use word-boundaries `\\b` for identifier searches."},
                "path_glob": {"type": "string", "description": "Optional path scope, e.g. `values/saas/**`."},
                "case_insensitive": {"type": "boolean", "description": "Default false."},
                "max_hits": {"type": "integer", "description": "Default 200, cap 200."},
            },
            "required": ["repo", "pattern"],
            "additionalProperties": False,
        },
        handler=_h_code_grep,
    ),
    MCPTool(
        name="code_read_file",
        description=(
            "Return a bounded line-range of a file from a repo (max 500 "
            "lines / 200KB per call). Use after `code_grep` to inspect "
            "context around a hit, or after `code_glob` when you've "
            "identified the relevant file. Don't dump entire files -- pick "
            "a line range around the symbol you care about."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "path": {"type": "string", "description": "Path relative to the repo root, e.g. `widget/views/widget_view.py`."},
                "start_line": {"type": "integer", "description": "1-indexed; default 1."},
                "end_line": {"type": "integer", "description": "1-indexed inclusive; default start_line + 499."},
            },
            "required": ["repo", "path"],
            "additionalProperties": False,
        },
        handler=_h_code_read_file,
    ),
    MCPTool(
        name="code_find_symbol",
        description=(
            "Find every file that DECLARES a symbol with this exact name "
            "(class / function / def / const / type) across one repo or all "
            "cached repos. Stricter than `code_grep` -- only matches "
            "declaration sites, not references. Use this when the user "
            "asks 'where is X defined' or 'which file owns class X'.\n\n"
            "Supported language kinds: python, typescript (covers .ts/.tsx/"
            ".js/.jsx), go, shell. Omit `kind` to search all languages.\n\n"
            "Examples:\n"
            "  - `name='WidgetViewSet', kind='python'` -- find the class def\n"
            "  - `name='getReportPdfUrl', kind='typescript'` -- find the FE method"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Identifier; must match `[A-Za-z_][A-Za-z0-9_]*`."},
                "kind": {"type": "string", "description": "Optional: python | typescript | go | shell."},
                "repo": {"type": "string", "description": "Optional: limit search to one repo."},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
        handler=_h_code_find_symbol,
    ),
    MCPTool(
        name="code_dependency_lookup",
        description=(
            "Read a repo's dependency manifests + lockfiles to answer dependency "
            "questions. Always prefer this over `code_grep` for anything about "
            "libraries/versions — it parses the LOCKFILE so it returns resolved/"
            "pinned versions, not just manifest ranges.\n\n"
            "TWO MODES:\n"
            "  1. LIST ALL (omit `package`): returns EVERY dependency with its "
            "version + ecosystem. Use this for 'what language / libraries / "
            "versions does <repo> use?' — ONE call answers it. Do NOT loop "
            "`code_grep` over manifest files; call this with just `repo`.\n"
            "  2. RESOLVE ONE (pass `package`): returns `resolved_version` + "
            "`declared_constraint` for that package, matched CASE-INSENSITIVELY "
            "('flask' finds 'Flask', 'react' finds 'React').\n\n"
            "Covers Python (pyproject.toml / poetry.lock / uv.lock / "
            "requirements*.txt), Node (package.json / package-lock.json / "
            "yarn.lock / pnpm-lock.yaml), Go (go.mod / go.sum), and Rust "
            "(Cargo.toml / Cargo.lock). For a monorepo, pass `path` to scope to "
            "a sub-directory.\n\n"
            "Examples:\n"
            "  - LIST: `repo='saas/acme-notes-be'`  -> all deps + versions\n"
            "  - ONE:  `repo='saas/acme-notes-be', package='django'`\n"
            "  - SCOPED: `repo='infra/mono', package='gin', path='services/api'`"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "Repo name. Call `code_list_repos` first if unsure."},
                "package": {"type": "string", "description": "Optional. Omit to LIST all dependencies; pass a name (case-insensitive) to resolve one."},
                "path": {"type": "string", "description": "Optional sub-directory to scope the scan (monorepos)."},
            },
            "required": ["repo"],
            "additionalProperties": False,
        },
        handler=_h_code_dependency_lookup,
    ),
]
