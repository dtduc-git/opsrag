"""R2 + R10 for opsrag.qa_cache.

R2 -- single-token service discriminator. The kebab-case extractor only
fires for >= 2-segment names, so single-token services ("auth", "billing",
"redis") used to produce IDENTICAL discriminator sets -> a "auth logs"
query could cache-hit a "billing logs" answer. The new
``_service_discriminator_regex`` (built from the active deployment's
service inventory) tags ``svc:<name>`` tokens so the two queries now
diverge and force a miss.

R10 -- repo invalidation matches the stored ground-truth ``repos`` field
written by ``store`` (with the legacy heuristic ``source_repos`` kept as a
fallback), not only the ``_derive_repos`` path heuristic.
"""
from __future__ import annotations

import asyncio

import pytest

from opsrag.agent.prompt_render import set_active_deployment
from opsrag.context import DeploymentContext
from opsrag.qa_cache import (
    QAVectorCache,
    _discriminator_tokens,
    _service_discriminator_regex,
)


@pytest.fixture(autouse=True)
def _reset_deployment():
    """Each test owns the process-global deployment context."""
    set_active_deployment(None)
    yield
    set_active_deployment(None)


# --------------------------------------------------------------------------- R2


def test_single_token_services_produce_different_discriminator_sets():
    """Two single-token services -> different svc: tokens -> forced miss.

    Without R2 these sets were equal (the kebab extractor never fires on a
    1-segment name), so a high-cosine match would wrongly serve the other
    service's cached answer."""
    set_active_deployment(DeploymentContext(services=["auth", "billing", "redis"]))

    auth = _discriminator_tokens("show me the auth logs")
    billing = _discriminator_tokens("show me the billing logs")

    assert "svc:auth" in auth
    assert "svc:billing" in billing
    # The discriminator sets MUST differ so lookup forces a cache miss.
    assert auth != billing


def test_single_token_service_no_inventory_is_no_op():
    """Empty service inventory -> regex is None -> no svc: tokens emitted
    (no regression for org-free deployments)."""
    set_active_deployment(None)
    assert _service_discriminator_regex() is None
    toks = _discriminator_tokens("show me the auth logs")
    assert not any(t.startswith("svc:") for t in toks)


def test_service_regex_is_whole_word_and_case_insensitive():
    set_active_deployment(DeploymentContext(services=["auth"]))
    # Substring inside another word must NOT match.
    assert not any(
        t == "svc:auth" for t in _discriminator_tokens("authenticate the user")
    )
    # Case-insensitive whole-word match does.
    assert "svc:auth" in _discriminator_tokens("check AUTH now")


def test_multi_segment_service_also_tagged_svc():
    """Multi-segment services get both kebab: and svc: tokens; two distinct
    multi-segment services still diverge."""
    set_active_deployment(
        DeploymentContext(services=["api-gateway", "kafka-broker"])
    )
    a = _discriminator_tokens("api-gateway latency")
    b = _discriminator_tokens("kafka-broker latency")
    assert "svc:api-gateway" in a
    assert "svc:kafka-broker" in b
    assert a != b


# -------------------------------------------------------------------------- R10


class _FakeDeleteClient:
    """Captures the filter passed to ``delete`` and stubs collection setup."""

    def __init__(self):
        self.delete_filter = None

    async def get_collections(self):
        class _C:
            collections = []
        return _C()

    async def create_collection(self, **kwargs):
        return None

    async def create_payload_index(self, **kwargs):
        return None

    async def delete(self, *, collection_name, points_selector, wait=False):
        self.delete_filter = points_selector.filter
        return None


def _condition_keys(flt) -> list[str]:
    """Pull FieldCondition keys out of a qdrant Filter's should clause."""
    return [c.key for c in (flt.should or [])]


def test_invalidate_repo_matches_stored_repos_field():
    """invalidate_repo's filter targets the ground-truth ``repos`` field
    (R10), keeping the heuristic ``source_repos`` as a fallback."""
    client = _FakeDeleteClient()
    cache = QAVectorCache(client, dimension=3)

    rc = asyncio.run(cache.invalidate_repo("owner/real-repo"))
    assert rc == -1  # qdrant doesn't return a count

    keys = _condition_keys(client.delete_filter)
    # Ground-truth field must be present; heuristic kept as fallback.
    assert "repos" in keys
    assert "source_repos" in keys
    # And both conditions match the requested repo value.
    for cond in client.delete_filter.should:
        assert cond.match.value == "owner/real-repo"


def test_store_writes_real_repos_into_repos_payload():
    """When the caller passes the real source_repos, store records them on
    the ``repos`` payload field (not just the path heuristic). The same
    entry is then reachable by invalidate_repo's ``repos`` match."""

    class _CapturingClient(_FakeDeleteClient):
        def __init__(self):
            super().__init__()
            self.upserted_payload = None

        async def upsert(self, *, collection_name, points, wait=False):
            self.upserted_payload = points[0].payload
            return None

    client = _CapturingClient()
    cache = QAVectorCache(client, dimension=3)

    asyncio.run(
        cache.store(
            question="how do I roll back the deploy",
            embedding=[0.0, 1.0, 0.0],
            answer="Run the rollback playbook; it reverts the last release safely.",
            sources=["docs/runbooks/rollback.md"],  # heuristic would mis-derive
            source_repos=["acme/payments-svc"],     # the REAL repo
        )
    )
    payload = client.upserted_payload
    assert payload is not None
    # Ground-truth repo recorded under the stored field invalidate matches.
    assert payload["repos"] == ["acme/payments-svc"]
    # Back-compat field carries the same real set.
    assert payload["source_repos"] == ["acme/payments-svc"]


def test_store_falls_back_to_heuristic_when_no_real_repos():
    """No caller-provided repos -> ``repos`` falls back to _derive_repos so
    invalidation still has something to match on for legacy call paths."""

    class _CapturingClient(_FakeDeleteClient):
        def __init__(self):
            super().__init__()
            self.upserted_payload = None

        async def upsert(self, *, collection_name, points, wait=False):
            self.upserted_payload = points[0].payload
            return None

    client = _CapturingClient()
    cache = QAVectorCache(client, dimension=3)

    asyncio.run(
        cache.store(
            question="what does the gateway service do",
            embedding=[0.0, 1.0, 0.0],
            answer="The gateway service terminates TLS and routes requests upstream.",
            # take=4/3 land on a filtered extension (.yaml) and are rejected,
            # so the heuristic backs off to the 2-segment owner/repo prefix.
            sources=["acme/api-gateway/config.yaml"],
        )
    )
    payload = client.upserted_payload
    assert payload is not None
    # Heuristic derived owner/repo prefix from the source path.
    assert payload["repos"] == ["acme/api-gateway"]
    assert payload["source_repos"] == payload["repos"]
