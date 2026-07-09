"""Config-driven high-priority repos: the retrieval boost target moves from a
hardcoded constant to `config.priority_repos`, bound once (build_providers) and
used as the single source of truth by BOTH the qdrant index-time payload tag and
the priority-module fallback derivation.
"""
from __future__ import annotations

from opsrag.config import Settings
from opsrag.vectorstores import priority as pr
from opsrag.vectorstores import qdrant as qd


def teardown_function():
    pr.set_priority_repos(None)  # restore default between tests


def test_config_default():
    assert Settings().priority_repos == ["sre-knowledge-base"]


def test_set_replaces_when_nonempty_else_keeps_default():
    pr.set_priority_repos(["my-kb", "  Runbooks-Repo  "])
    assert pr.high_priority_repo_substr() == ("my-kb", "runbooks-repo")  # trimmed+lowered
    pr.set_priority_repos([])            # empty -> default
    assert pr.high_priority_repo_substr() == ("sre-knowledge-base",)
    pr.set_priority_repos(None)          # None -> default
    assert pr.high_priority_repo_substr() == ("sre-knowledge-base",)


def test_chunk_priority_respects_config_both_stores():
    # A repo that ISN'T matched by the default substring.
    repo = "org/team/internal-runbooks"
    assert pr.chunk_priority(repo, None) is None     # default set doesn't match
    assert qd._chunk_priority(repo, None) is None

    pr.set_priority_repos(["internal-runbooks"])    # config it in
    assert pr.chunk_priority(repo, None) == "high"    # fallback derivation
    assert qd._chunk_priority(repo, None) == "high"   # index-time payload tag
    # architecture path -> higher tier, from both
    arch = "docs/architecture/topology.md"
    assert pr.chunk_priority(repo, arch) == "architecture-canonical"
    assert qd._chunk_priority(repo, arch) == "architecture-canonical"

    # a non-listed repo stays unboosted
    assert qd._chunk_priority("saas/some-service", None) is None


def test_build_providers_binds(monkeypatch):
    """build_providers should push config.priority_repos into the module."""
    import opsrag.factory as factory
    seen = {}
    monkeypatch.setattr(pr, "set_priority_repos", lambda r: seen.setdefault("r", r))
    # stop build_providers before the heavy provider wiring — we only need the
    # early bind line to have run.
    monkeypatch.setattr(factory, "_env", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("stop")))
    s = Settings()
    s.priority_repos = ["kb-a", "kb-b"]
    try:
        factory.build_providers(s)
    except RuntimeError:
        pass
    assert seen.get("r") == ["kb-a", "kb-b"]
