"""Security regression test (TRACK rce / C2): git-grep argument-injection RCE.

The LLM-controlled `pattern` was appended to the `git grep` argv with no `-e`
flag and no `--` separator. `--full-name` does NOT stop git's option parsing,
so a pattern like::

    --open-files-in-pager=touch /tmp/x;true

was interpreted as a git option and executed an arbitrary command. The fix
binds the pattern to `-e` and places `--` before any path arg, and (defense in
depth) rejects patterns that start with '-' in code_grep.

These tests run the REAL `git grep` subprocess against the offline fake repo
(build_fake stands up a throwaway git repo, no network). Coverage:
- the leading-dash `pattern` is refused by the Python guard before any git runs
  (proves the guard, not the argv binding);
- a malicious `path_glob` (the ACTUAL reproduced vector) is inert because it now
  sits after `--`; and
- a parity test reconstructs the pre-fix argv to confirm the old form fired the
  pager and the fixed form does not.

asyncio_mode = "auto".
"""
from __future__ import annotations

import pytest

from opsrag.mcp.code import build_fake

_PAGER_INJECTION = "--open-files-in-pager=touch {sentinel};true"
_REPO = "group/example-repo"


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()


async def test_grep_rejects_leading_dash_pattern_no_rce(fake, tmp_path) -> None:
    """A pattern starting with '--open-files-in-pager=...' must be rejected as
    a literal-ish bad pattern, NOT executed as a git option (no RCE)."""
    sentinel = tmp_path / "pwned_grep"
    assert not sentinel.exists()

    result = await fake.call(
        "code_grep",
        {"repo": _REPO, "pattern": _PAGER_INJECTION.format(sentinel=sentinel)},
    )

    # Defense-in-depth: leading-dash patterns are refused with a clear error.
    assert "error" in result
    assert result["error"].startswith("`pattern` must not start with '-'")
    # The decisive check: the injected command never ran.
    assert not sentinel.exists(), "argument injection executed -- RCE still live!"


async def test_grep_dashed_value_is_literal_when_not_leading(fake, tmp_path) -> None:
    """A pattern that merely CONTAINS a dangerous-looking option (but doesn't
    start with '-') is treated as a literal grep pattern via `-e`, never an
    option. No command executes; it simply finds no match in the fake repo."""
    sentinel = tmp_path / "pwned_embedded"
    # Leading 'x' so it doesn't trip the leading-dash guard; the rest mimics
    # the injection payload to prove `-e` binding stops option parsing.
    pattern = f"x--open-files-in-pager=touch {sentinel};true"

    result = await fake.call("code_grep", {"repo": _REPO, "pattern": pattern})

    assert "error" not in result
    assert result["count"] == 0  # literal pattern matches nothing in the seed repo
    assert not sentinel.exists(), "embedded option string executed as a git option!"


async def test_grep_e_binding_behavior_unchanged(fake) -> None:
    """behavior_change=none: a normal pattern still returns its expected hit
    now that it's bound to `-e ... --`."""
    result = await fake.call("code_grep", {"repo": _REPO, "pattern": "handle_request"})
    assert "error" not in result
    assert result["count"] >= 1
    assert "src/app.py" in {h["path"] for h in result["hits"]}


async def test_grep_path_glob_still_scopes_after_separator(fake) -> None:
    """The `--` now precedes the path glob; path scoping must still work."""
    scoped = await fake.call(
        "code_grep",
        {"repo": _REPO, "pattern": "func", "path_glob": "*.go"},
    )
    assert "error" not in scoped
    assert all(h["path"].endswith(".go") for h in scoped["hits"])
    assert scoped["count"] >= 1  # src/util.go has `func Add`


async def test_grep_path_glob_injection_no_rce(fake, tmp_path) -> None:
    """The ACTUAL exploitable vector: a benign MATCHING pattern plus a malicious
    path_glob carrying --open-files-in-pager. path_glob now sits AFTER `--`, so
    it is an inert pathspec, not a git option -- the pager must never fire."""
    sentinel = tmp_path / "pwned_pathglob"
    assert not sentinel.exists()
    result = await fake.call(
        "code_grep",
        {
            "repo": _REPO,
            "pattern": "handle_request",  # real match -> output exists
            "path_glob": _PAGER_INJECTION.format(sentinel=sentinel),
        },
    )
    assert "error" not in result  # path_glob is a valid (if odd) pathspec
    assert not sentinel.exists(), "path_glob argument injection executed -- RCE still live!"


async def test_grep_old_argv_was_vulnerable_new_is_safe(fake, tmp_path) -> None:
    """Parity proof: reconstruct the PRE-FIX argv (path_glob option-parsed before
    any `--`) vs the fixed argv. If this git build reproduces the pager exploit,
    the OLD form fires the sentinel and the FIXED form does not. Skips if the
    local git doesn't honor --open-files-in-pager (version-dependent)."""
    from opsrag.mcp.code import _repo_dir, _run_git

    d = _repo_dir(_REPO)
    assert d is not None
    old_sent = tmp_path / "old_fires"
    new_sent = tmp_path / "new_safe"
    # PRE-FIX argv: pattern + path_glob both positional -> path_glob option-parsed.
    await _run_git(
        ["grep", "-n", "-E", "--full-name", "handle_request",
         f"--open-files-in-pager=touch {old_sent};true"],
        cwd=d, timeout=4.0,
    )
    if not old_sent.exists():
        pytest.skip("local git does not reproduce the --open-files-in-pager exploit")
    # FIXED argv: pattern bound to -e, path_glob AFTER `--` (inert pathspec).
    await _run_git(
        ["grep", "-n", "-E", "--full-name", "-e", "handle_request", "--",
         f"--open-files-in-pager=touch {new_sent};true"],
        cwd=d, timeout=4.0,
    )
    assert not new_sent.exists(), "fixed argv STILL fires the pager -- RCE not closed!"


async def test_find_symbol_rejects_option_like_name_no_rce(fake, tmp_path) -> None:
    """code_find_symbol already validates `name` to an identifier; an
    option-looking name is rejected before any subprocess runs (no RCE)."""
    sentinel = tmp_path / "pwned_symbol"
    result = await fake.call(
        "code_find_symbol",
        {"name": f"--open-files-in-pager=touch {sentinel};true"},
    )
    assert "error" in result
    assert "single identifier" in result["error"]
    assert not sentinel.exists()


async def test_find_symbol_behavior_unchanged(fake) -> None:
    """behavior_change=none: binding the regex to `-e ... --` returns exactly
    the same hits the bare-arg form would.

    We assert parity against a direct `git grep` run that reproduces the
    pre-fix argv (bare pattern, no `-e`). Some POSIX-ERE git builds reject the
    `(?:...)` groups in the symbol regex; whatever the engine does, the new and
    old argv must agree -- that's the real guarantee. (No `_GREP_MAX_HITS`
    truncation concern for this tiny seed repo.)
    """
    from opsrag.mcp.code import _build_symbol_regex, _repo_dir, _run_git

    regex, globs = _build_symbol_regex("handle_request", "python")
    d = _repo_dir(_REPO)
    assert d is not None

    new_args = ["grep", "-n", "-E", "--full-name", "-e", regex, "--", *globs]
    old_args = ["grep", "-n", "-E", "--full-name", regex, "--", *globs]
    new_code, new_out, _ = await _run_git(new_args, cwd=d, timeout=3.0)
    old_code, old_out, _ = await _run_git(old_args, cwd=d, timeout=3.0)
    assert (new_code, new_out) == (old_code, old_out)

    # And the handler runs cleanly (no error key) regardless of engine support.
    result = await fake.call("code_find_symbol", {"name": "handle_request", "kind": "python"})
    assert "error" not in result
    assert isinstance(result["hits"], list)
