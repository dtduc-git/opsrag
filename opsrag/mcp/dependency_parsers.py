"""Pure parsers for dependency manifests + lockfiles (NO I/O, NO clone).

Given file *contents*, extract a package's declared constraint (from the
manifest) and resolved version (from the lockfile). Name matching is
CASE-INSENSITIVE (and PEP 503-normalized for Python: ``Flask`` == ``flask``).

This is the logic the ``code_dependency_lookup`` MCP tool needs that plain
``code_grep`` lacks: (1) it reads the LOCKFILE so it returns the *resolved*
version, not just the manifest's range, and (2) it matches names
case-insensitively. Kept pure so it unit-tests without a repo on disk.

Ecosystems: Python (pyproject.toml / poetry.lock / uv.lock / requirements*.txt),
Node (package.json / package-lock.json / yarn.lock / pnpm-lock.yaml),
Go (go.mod / go.sum), Rust (Cargo.toml / Cargo.lock).
"""
from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Callable
from typing import Any

_MAX_FINDINGS = 60


# --- name normalization ----------------------------------------------------

def _norm_py(name: str) -> str:
    """PEP 503 normalized name: lowercase, runs of -_. collapsed to a single -."""
    return re.sub(r"[-_.]+", "-", (name or "").strip()).lower()


def _norm(name: str) -> str:
    return (name or "").strip().lower()


# --- Python ----------------------------------------------------------------

def _split_pep508(spec: str) -> tuple[str | None, str]:
    spec = (spec or "").strip()
    m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*([^;]*)", spec)
    if not m:
        return None, "*"
    return m.group(1), (m.group(3) or "").strip() or "*"


def _poetry_constraint(val: Any) -> str:
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return str(val.get("version") or val.get("git") or val.get("path") or val.get("url") or val)
    return str(val)


def parse_pyproject(text: str) -> list[dict]:
    data = tomllib.loads(text)
    out: list[dict] = []
    proj = data.get("project") or {}
    for spec in proj.get("dependencies") or []:
        nm, cons = _split_pep508(spec)
        if nm:
            out.append({"name": nm, "version": cons, "section": "project.dependencies"})
    for grp, specs in (proj.get("optional-dependencies") or {}).items():
        for spec in specs or []:
            nm, cons = _split_pep508(spec)
            if nm:
                out.append({"name": nm, "version": cons, "section": f"optional-dependencies.{grp}"})
    for grp, specs in (data.get("dependency-groups") or {}).items():  # PEP 735
        for spec in specs or []:
            if isinstance(spec, str):
                nm, cons = _split_pep508(spec)
                if nm:
                    out.append({"name": nm, "version": cons, "section": f"dependency-groups.{grp}"})
    poetry = (data.get("tool") or {}).get("poetry") or {}
    for key in ("dependencies", "dev-dependencies"):
        for nm, val in (poetry.get(key) or {}).items():
            if nm.lower() == "python":
                continue
            out.append({"name": nm, "version": _poetry_constraint(val), "section": f"poetry.{key}"})
    for grp, gd in (poetry.get("group") or {}).items():
        for nm, val in ((gd or {}).get("dependencies") or {}).items():
            if nm.lower() == "python":
                continue
            out.append({"name": nm, "version": _poetry_constraint(val), "section": f"poetry.group.{grp}"})
    return out


def parse_requirements(text: str) -> list[dict]:
    out: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        line = line.split(";")[0].split("#")[0].strip()
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*(.*)$", line)
        if not m:
            continue
        out.append({"name": m.group(1), "version": (m.group(3) or "").strip() or "*"})
    return out


def parse_toml_lock(text: str) -> list[dict]:
    """poetry.lock / uv.lock / Cargo.lock -- ``[[package]]`` array of {name, version}."""
    data = tomllib.loads(text)
    return [
        {"name": p.get("name"), "version": p.get("version")}
        for p in (data.get("package") or [])
        if p.get("name") and p.get("version")
    ]


# --- Node ------------------------------------------------------------------

def parse_package_json(text: str) -> list[dict]:
    data = json.loads(text)
    out: list[dict] = []
    for sect in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
        for nm, rng in (data.get(sect) or {}).items():
            out.append({"name": nm, "version": rng, "section": sect})
    return out


def parse_package_lock(text: str) -> list[dict]:
    data = json.loads(text)
    out: list[dict] = []
    pkgs = data.get("packages")
    if isinstance(pkgs, dict):  # lockfileVersion 2/3
        for key, meta in pkgs.items():
            if "node_modules/" not in key:
                continue
            nm = key.split("node_modules/")[-1]  # last segment handles nesting
            ver = (meta or {}).get("version")
            if nm and ver:
                out.append({"name": nm, "version": ver})
    deps = data.get("dependencies")
    if isinstance(deps, dict):  # lockfileVersion 1
        def _walk(d: dict) -> None:
            for nm, meta in d.items():
                ver = (meta or {}).get("version")
                if ver:
                    out.append({"name": nm, "version": ver})
                nested = (meta or {}).get("dependencies")
                if isinstance(nested, dict):
                    _walk(nested)
        _walk(deps)
    return out


def parse_yarn_lock(text: str) -> list[dict]:
    out: list[dict] = []
    cur_names: list[str] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" ") and line.rstrip().endswith(":"):
            cur_names = []
            for spec in line.rstrip()[:-1].split(","):
                spec = spec.strip().strip('"')
                if spec.startswith("@"):  # scoped: @scope/pkg@range
                    at = spec.rfind("@")
                    nm = spec[:at] if at > 0 else spec
                else:
                    nm = spec.split("@")[0]
                if nm:
                    cur_names.append(nm)
        else:
            m = re.match(r'\s+version:?\s+"?([^"\s]+)"?', line)
            if m and cur_names:
                for nm in cur_names:
                    out.append({"name": nm, "version": m.group(1)})
                cur_names = []
    return out


def parse_pnpm_lock(text: str) -> list[dict]:
    # Best-effort across pnpm-lock versions: capture `<name>@<version>` keys
    # (optionally scoped, optionally `/`-prefixed) in the packages section.
    out: list[dict] = []
    for m in re.finditer(r"^\s+'?/?(@?[\w.-]+(?:/[\w.-]+)?)@([0-9][\w.\-+]*)", text, re.M):
        out.append({"name": m.group(1), "version": m.group(2)})
    return out


# --- Go / Rust -------------------------------------------------------------

def parse_go_mod(text: str) -> list[dict]:
    out: list[dict] = []
    for m in re.finditer(r"^\s*([\w.\-]+(?:\.[\w.\-]+)?(?:/[\w.\-~]+)+)\s+(v[0-9]\S*)", text, re.M):
        out.append({"name": m.group(1), "version": m.group(2)})
    return out


def parse_go_sum(text: str) -> list[dict]:
    out: list[dict] = []
    for m in re.finditer(r"^(\S+)\s+(v[0-9]\S*?)(?:/go\.mod)?\s+h1:", text, re.M):
        out.append({"name": m.group(1), "version": m.group(2)})
    return out


def parse_cargo_toml(text: str) -> list[dict]:
    data = tomllib.loads(text)
    out: list[dict] = []
    for sect in ("dependencies", "dev-dependencies", "build-dependencies"):
        for nm, val in (data.get(sect) or {}).items():
            if isinstance(val, str):
                ver = val
            elif isinstance(val, dict):
                ver = val.get("version") or str(val)
            else:
                ver = str(val)
            out.append({"name": nm, "version": ver, "section": sect})
    return out


# --- dispatch + orchestrator ----------------------------------------------

# basename -> (ecosystem, default_role, parser). ``requirements`` is matched
# by prefix below (requirements*.txt). Role for requirements is decided
# per-entry (== pin => resolved).
_DISPATCH: dict[str, tuple[str, str, Callable[[str], list[dict]]]] = {
    "pyproject.toml": ("python", "declared", parse_pyproject),
    "poetry.lock": ("python", "resolved", parse_toml_lock),
    "uv.lock": ("python", "resolved", parse_toml_lock),
    "package.json": ("node", "declared", parse_package_json),
    "package-lock.json": ("node", "resolved", parse_package_lock),
    "yarn.lock": ("node", "resolved", parse_yarn_lock),
    "pnpm-lock.yaml": ("node", "resolved", parse_pnpm_lock),
    "go.mod": ("go", "declared", parse_go_mod),
    "go.sum": ("go", "resolved", parse_go_sum),
    "Cargo.toml": ("rust", "declared", parse_cargo_toml),
    "Cargo.lock": ("rust", "resolved", parse_toml_lock),
}

# All filenames the lookup tool should collect from a repo.
DEP_FILE_NAMES: frozenset[str] = frozenset(_DISPATCH)


def is_dependency_file(basename: str) -> bool:
    if basename in _DISPATCH:
        return True
    return basename.startswith("requirements") and basename.endswith(".txt")


def _matches(ecosystem: str, dep_name: str, pkg: str) -> bool:
    if not dep_name:
        return False
    if ecosystem == "python":
        return _norm_py(dep_name) == _norm_py(pkg)
    if ecosystem == "go":
        dn, pk = _norm(dep_name), _norm(pkg)
        # accept full module path OR a trailing path segment (e.g. "gin").
        return dn == pk or dn.endswith("/" + pk) or pk in dn.split("/")
    return _norm(dep_name) == _norm(pkg)


def resolve_dependency(files: dict[str, str], package: str) -> dict:
    """Search ``files`` ({relpath: content}) for ``package``. Returns the
    resolved version (lockfile) + declared constraint (manifest) + every
    matching finding. Never raises on a malformed file."""
    pkg = (package or "").strip()
    findings: list[dict] = []
    for relpath, text in files.items():
        base = relpath.rsplit("/", 1)[-1]
        if base in _DISPATCH:
            ecosystem, role, parser = _DISPATCH[base]
        elif base.startswith("requirements") and base.endswith(".txt"):
            ecosystem, role, parser = "python", "requirements", parse_requirements
        else:
            continue
        try:
            entries = parser(text)
        except Exception:
            continue  # malformed file -> skip, never propagate
        for e in entries:
            if not _matches(ecosystem, e.get("name", ""), pkg):
                continue
            ver = e.get("version") or "*"
            r = role
            if role == "requirements":
                sver = str(ver).strip()
                if sver.startswith("=="):
                    r = "resolved"
                    ver = sver[2:].strip()  # clean pinned version
                else:
                    r = "declared"
            findings.append({
                "ecosystem": ecosystem,
                "file": relpath,
                "role": r,
                "name": e.get("name"),
                "version": ver,
                **({"section": e["section"]} if e.get("section") else {}),
            })
            if len(findings) >= _MAX_FINDINGS:
                break
        if len(findings) >= _MAX_FINDINGS:
            break

    resolved = next((f["version"] for f in findings if f["role"] == "resolved"), None)
    declared = next((f["version"] for f in findings if f["role"] == "declared"), None)
    ecosystems = sorted({f["ecosystem"] for f in findings})
    return {
        "package": pkg,
        "found": bool(findings),
        "resolved_version": resolved,
        "declared_constraint": declared,
        "ecosystems": ecosystems,
        "findings": findings,
    }


def detect_languages(files: dict[str, str]) -> dict:
    """Extract the LANGUAGE/runtime version each ecosystem declares (distinct
    from library deps). Answers "what language version does X use?".

    Python  -> poetry ``[tool.poetry.dependencies] python`` or PEP 621
               ``[project] requires-python``.
    Go      -> the ``go X.Y`` directive in go.mod.
    Node    -> ``engines.node`` in package.json.
    Rust    -> ``package.rust-version`` (falls back to ``edition``) in Cargo.toml.
    Returns ``{language: version_constraint}``; never raises."""
    langs: dict[str, str] = {}
    for relpath, text in files.items():
        base = relpath.rsplit("/", 1)[-1]
        try:
            if base == "pyproject.toml":
                data = tomllib.loads(text)
                poetry = (data.get("tool") or {}).get("poetry") or {}
                py = (poetry.get("dependencies") or {}).get("python")
                if py:
                    langs.setdefault("python", _poetry_constraint(py))
                else:
                    rp = (data.get("project") or {}).get("requires-python")
                    if rp:
                        langs.setdefault("python", str(rp).strip())
            elif base == "go.mod":
                m = re.search(r"^go\s+(\d+(?:\.\d+){1,2})", text, re.MULTILINE)
                if m:
                    langs.setdefault("go", m.group(1))
            elif base == "package.json":
                node = ((json.loads(text).get("engines") or {}).get("node"))
                if node:
                    langs.setdefault("node", str(node).strip())
            elif base == "Cargo.toml":
                pkg = (tomllib.loads(text).get("package") or {})
                rv = pkg.get("rust-version") or pkg.get("edition")
                if rv:
                    langs.setdefault("rust", str(rv).strip())
        except Exception:
            continue  # malformed file -> skip, never propagate
    return langs


def list_dependencies(files: dict[str, str], max_deps: int = 400) -> dict:
    """Parse every manifest/lockfile in ``files`` and return the FULL dependency
    set (not a single package). Answers "what libraries/versions does X use?".

    Per (ecosystem, normalized-name) we keep the most authoritative version: a
    lockfile-resolved pin wins over a manifest constraint. Never raises on a
    malformed file."""
    # key -> dict(name, version, role, ecosystem, file, section?)
    best: dict[tuple[str, str], dict] = {}
    _rank = {"resolved": 2, "declared": 1, "constraint": 1}
    for relpath, text in files.items():
        base = relpath.rsplit("/", 1)[-1]
        if base in _DISPATCH:
            ecosystem, role, parser = _DISPATCH[base]
        elif base.startswith("requirements") and base.endswith(".txt"):
            ecosystem, role, parser = "python", "requirements", parse_requirements
        else:
            continue
        try:
            entries = parser(text)
        except Exception:
            continue  # malformed file -> skip, never propagate
        for e in entries:
            name = e.get("name")
            if not name:
                continue
            ver = e.get("version") or "*"
            r = role
            if role == "requirements":
                sver = str(ver).strip()
                if sver.startswith("=="):
                    r, ver = "resolved", sver[2:].strip()
                else:
                    r = "declared"
            nkey = _norm_py(name) if ecosystem == "python" else _norm(name)
            key = (ecosystem, nkey)
            entry = {
                "ecosystem": ecosystem,
                "name": name,
                "version": ver,
                "role": r,
                "file": relpath,
                **({"section": e["section"]} if e.get("section") else {}),
            }
            cur = best.get(key)
            if cur is None or _rank.get(r, 0) > _rank.get(cur["role"], 0):
                best[key] = entry

    deps = sorted(best.values(), key=lambda d: (d["ecosystem"], d["name"].lower()))
    truncated = len(deps) > max_deps
    if truncated:
        deps = deps[:max_deps]
    return {
        "found": bool(deps),
        "count": len(deps),
        "ecosystems": sorted({d["ecosystem"] for d in deps}),
        "languages": detect_languages(files),
        "dependencies": deps,
        **({"truncated_to": max_deps} if truncated else {}),
    }
