"""Tests for the dependency_lookup feature: the pure parsers/orchestrator
(opsrag.mcp.dependency_parsers) + the code_dependency_lookup MCP handler
(via the offline code fake). Covers the two gaps that made plain code_grep
fail on flask/react: lockfile resolution + case-insensitive matching.
asyncio_mode = "auto"."""
from __future__ import annotations

import pytest

from opsrag.mcp.dependency_parsers import (
    is_dependency_file,
    resolve_dependency,
)

# --- the flask scenario: pyproject range + poetry.lock pin, case mismatch --


def test_python_resolves_from_lockfile_case_insensitive():
    files = {
        "pyproject.toml": '[project]\ndependencies = ["Flask>=2.3", "requests"]\n',
        "poetry.lock": '[[package]]\nname = "flask"\nversion = "2.3.3"\n',
    }
    # User asks for lowercase 'flask'; manifest has 'Flask'; lockfile has 'flask'.
    r = resolve_dependency(files, "flask")
    assert r["found"] is True
    assert r["resolved_version"] == "2.3.3"          # from the lockfile
    assert r["declared_constraint"] == ">=2.3"        # from the manifest
    assert "python" in r["ecosystems"]
    roles = {f["role"] for f in r["findings"]}
    assert roles == {"declared", "resolved"}


def test_python_pep503_normalization():
    # 'ruamel.yaml' / 'ruamel-yaml' / 'ruamel_yaml' are the same package.
    files = {"uv.lock": '[[package]]\nname = "ruamel-yaml"\nversion = "0.18.6"\n'}
    assert resolve_dependency(files, "ruamel.yaml")["resolved_version"] == "0.18.6"
    assert resolve_dependency(files, "RUAMEL_YAML")["resolved_version"] == "0.18.6"


def test_poetry_table_constraint_and_groups():
    files = {
        "pyproject.toml": (
            "[tool.poetry.dependencies]\n"
            'python = "^3.11"\n'
            'django = { version = "^5.0", extras = ["argon2"] }\n'
            "[tool.poetry.group.dev.dependencies]\n"
            'pytest = "^8.0"\n'
        ),
    }
    assert resolve_dependency(files, "django")["declared_constraint"] == "^5.0"
    assert resolve_dependency(files, "pytest")["declared_constraint"] == "^8.0"
    assert resolve_dependency(files, "python")["found"] is False  # python is skipped


def test_requirements_pin_is_resolved():
    files = {"requirements.txt": "Flask==2.3.3\nrequests>=2.0\n# comment\n-r other.txt\n"}
    assert resolve_dependency(files, "flask")["resolved_version"] == "2.3.3"
    # range-only dep -> declared, no resolved
    req = resolve_dependency(files, "requests")
    assert req["declared_constraint"] == ">=2.0"
    assert req["resolved_version"] is None


# --- the react scenario: package.json range + package-lock pin -------------


def test_node_resolves_from_package_lock():
    files = {
        "package.json": '{"dependencies": {"react": "^18.2.0"}}',
        "package-lock.json": (
            '{"lockfileVersion": 3, "packages": '
            '{"": {"name": "x"}, "node_modules/react": {"version": "18.2.0"}}}'
        ),
    }
    r = resolve_dependency(files, "React")  # case-insensitive
    assert r["resolved_version"] == "18.2.0"
    assert r["declared_constraint"] == "^18.2.0"


def test_node_package_lock_v1_and_nested():
    files = {
        "package-lock.json": (
            '{"lockfileVersion": 1, "dependencies": '
            '{"lodash": {"version": "4.17.21", "dependencies": '
            '{"nested-dep": {"version": "1.0.0"}}}}}'
        ),
    }
    assert resolve_dependency(files, "lodash")["resolved_version"] == "4.17.21"
    assert resolve_dependency(files, "nested-dep")["resolved_version"] == "1.0.0"


def test_yarn_lock_scoped_and_plain():
    files = {
        "yarn.lock": (
            'lodash@^4.17.21:\n  version "4.17.21"\n  resolved "..."\n\n'
            '"@babel/core@^7.0.0", "@babel/core@^7.1.0":\n  version "7.23.0"\n'
        ),
    }
    assert resolve_dependency(files, "lodash")["resolved_version"] == "4.17.21"
    assert resolve_dependency(files, "@babel/core")["resolved_version"] == "7.23.0"


def test_pnpm_lock_best_effort():
    files = {"pnpm-lock.yaml": "packages:\n  /react@18.2.0:\n    resolution: {integrity: sha}\n"}
    assert resolve_dependency(files, "react")["resolved_version"] == "18.2.0"


# --- go + rust -------------------------------------------------------------


def test_go_mod_full_path_and_suffix():
    files = {"go.mod": "module x\n\nrequire (\n\tgithub.com/gin-gonic/gin v1.9.1\n)\n"}
    assert resolve_dependency(files, "github.com/gin-gonic/gin")["declared_constraint"] == "v1.9.1"
    # suffix segment match
    assert resolve_dependency(files, "gin")["declared_constraint"] == "v1.9.1"


def test_cargo_resolves_from_lock():
    files = {
        "Cargo.toml": '[dependencies]\nserde = "1.0"\ntokio = { version = "1.35" }\n',
        "Cargo.lock": '[[package]]\nname = "serde"\nversion = "1.0.195"\n',
    }
    r = resolve_dependency(files, "serde")
    assert r["resolved_version"] == "1.0.195"
    assert r["declared_constraint"] == "1.0"
    assert resolve_dependency(files, "tokio")["declared_constraint"] == "1.35"


# --- robustness ------------------------------------------------------------


def test_not_found_and_malformed_files_dont_raise():
    files = {
        "pyproject.toml": "this is { not valid toml [[[",
        "package.json": "{not json",
        "poetry.lock": '[[package]]\nname = "flask"\nversion = "2.3.3"\n',
    }
    # malformed manifest/json are skipped; valid lockfile still resolves.
    assert resolve_dependency(files, "flask")["resolved_version"] == "2.3.3"
    assert resolve_dependency(files, "nonexistent-pkg")["found"] is False


def test_list_dependencies_full_set_lockfile_wins():
    from opsrag.mcp.dependency_parsers import list_dependencies

    files = {
        "pyproject.toml": '[project]\ndependencies = ["Flask>=2.3", "requests"]\n',
        "poetry.lock": (
            '[[package]]\nname = "flask"\nversion = "2.3.3"\n\n'
            '[[package]]\nname = "requests"\nversion = "2.31.0"\n'
        ),
        "package.json": '{"dependencies": {"react": "^18.2.0"}}',
    }
    res = list_dependencies(files)
    assert res["found"] is True
    by_name = {d["name"].lower(): d for d in res["dependencies"]}
    # lockfile-resolved version wins over the manifest constraint
    assert by_name["flask"]["version"] == "2.3.3"
    assert by_name["flask"]["role"] == "resolved"
    assert by_name["requests"]["version"] == "2.31.0"
    assert "react" in by_name
    assert set(res["ecosystems"]) >= {"python", "node"}


def test_detect_languages_python_go_node_rust():
    from opsrag.mcp.dependency_parsers import detect_languages

    files = {
        "pyproject.toml": (
            "[tool.poetry.dependencies]\npython = \"~3.12\"\ndjango = \"5.2.14\"\n"
        ),
        "go.mod": "module example.com/m\n\ngo 1.22\n\nrequire github.com/gin-gonic/gin v1.9.1\n",
        "package.json": '{"engines": {"node": ">=20"}, "dependencies": {"react": "^18"}}',
        "Cargo.toml": '[package]\nname = "x"\nrust-version = "1.75"\nedition = "2021"\n',
    }
    langs = detect_languages(files)
    assert langs["python"] == "~3.12"
    assert langs["go"] == "1.22"
    assert langs["node"] == ">=20"
    assert langs["rust"] == "1.75"


def test_list_dependencies_includes_languages():
    from opsrag.mcp.dependency_parsers import list_dependencies

    files = {"pyproject.toml": '[tool.poetry.dependencies]\npython = "~3.12"\ndjango = "5.2.14"\n'}
    res = list_dependencies(files)
    assert res["languages"].get("python") == "~3.12"
    assert any(d["name"].lower() == "django" for d in res["dependencies"])


def test_is_dependency_file():
    for ok in ("pyproject.toml", "poetry.lock", "uv.lock", "package.json",
               "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "go.mod",
               "go.sum", "Cargo.toml", "Cargo.lock", "requirements.txt",
               "requirements-dev.txt"):
        assert is_dependency_file(ok), ok
    for no in ("setup.py", "README.md", "app.py", "Makefile"):
        assert not is_dependency_file(no), no


# --- handler path (end-to-end over the offline code fake repo) -------------


@pytest.fixture
def code_fake():
    from opsrag.mcp.code import build_fake
    f = build_fake()
    try:
        yield f
    finally:
        f.close()


async def test_handler_resolves_python_dep(code_fake):
    res = await code_fake.call(
        "code_dependency_lookup", {"repo": "group/example-repo", "package": "flask"}
    )
    assert res["found"] is True
    assert res["resolved_version"] == "2.3.3"
    assert res["declared_constraint"] == ">=2.3"
    assert "pyproject.toml" in res["searched_files"]
    assert "poetry.lock" in res["searched_files"]


async def test_handler_resolves_node_dep_in_subdir(code_fake):
    res = await code_fake.call(
        "code_dependency_lookup", {"repo": "group/example-repo", "package": "React"}
    )
    assert res["resolved_version"] == "18.2.0"
    assert any(f.endswith("package-lock.json") for f in res["searched_files"])


async def test_handler_requires_repo(code_fake):
    res = await code_fake.call("code_dependency_lookup", {"package": "flask"})
    assert "error" in res


async def test_handler_lists_all_when_no_package(code_fake):
    """Omitting `package` lists EVERY dependency (the 'what libs does X use?'
    path) instead of erroring."""
    res = await code_fake.call("code_dependency_lookup", {"repo": "group/example-repo"})
    assert "error" not in res
    assert res["found"] is True
    assert res["count"] >= 1
    names = {d["name"].lower() for d in res["dependencies"]}
    assert "flask" in names


async def test_handler_not_found_gives_note(code_fake):
    res = await code_fake.call(
        "code_dependency_lookup", {"repo": "group/example-repo", "package": "no-such-pkg"}
    )
    assert res["found"] is False
    assert "note" in res
