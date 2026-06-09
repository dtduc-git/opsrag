"""Unit tests for the generalized Elasticsearch/OpenSearch MCP tools against
the offline fake backend (no network, no ES creds). asyncio_mode = "auto"."""
from __future__ import annotations

import pytest

from opsrag.mcp.elasticsearch import (
    ES_TOOLS,
    MCPElasticsearchError,
    build_fake,
    get_tool,
)

_EXPECTED_TOOLS = {
    "elasticsearch_list_indices",
    "elasticsearch_get_mappings",
    "elasticsearch_search",
    "elasticsearch_esql_query",
    "elasticsearch_cluster_health",
}


@pytest.fixture
def fake():
    f = build_fake()
    try:
        yield f
    finally:
        f.close()


def test_tool_set_matches_exactly(fake) -> None:
    assert set(fake.tool_names()) == _EXPECTED_TOOLS
    assert {t.name for t in ES_TOOLS} == _EXPECTED_TOOLS


def test_all_tools_are_read_only() -> None:
    # No tool name implies a mutation.
    mutating = ("create", "update", "delete", "put", "index_doc", "bulk", "_update", "reindex")
    for t in ES_TOOLS:
        assert not any(m in t.name for m in mutating), t.name


async def test_list_indices_hides_system_indices(fake) -> None:
    res = await fake.call("elasticsearch_list_indices", {})
    names = {i["index"] for i in res["indices"]}
    assert "app-logs-000001" in names
    assert not any(n.startswith(".") for n in names)  # .kibana_1 filtered out
    assert res["count"] == len(res["indices"])


async def test_get_mappings_flattens_field_types(fake) -> None:
    res = await fake.call("elasticsearch_get_mappings", {"index": "app-logs-000001"})
    fields = res["mappings"]["app-logs-000001"]
    assert fields["level"] == "keyword"
    assert fields["message"] == "text"


async def test_search_with_lucene_q(fake) -> None:
    res = await fake.call("elasticsearch_search", {"q": "level:error", "size": 5})
    assert res["count"] == 2
    assert res["hits"][0]["source"]["service"] == "payments"


async def test_search_with_dsl_query(fake) -> None:
    res = await fake.call(
        "elasticsearch_search",
        {"query": {"term": {"level": "error"}}, "index": "app-logs-000001"},
    )
    assert res["index"] == "app-logs-000001"
    assert res["count"] == 2


async def test_esql_query(fake) -> None:
    res = await fake.call(
        "elasticsearch_esql_query",
        {"query": 'FROM app-logs-* | STATS count() BY service'},
    )
    assert res["columns"] == ["service", "count()"]
    assert res["row_count"] == 2


async def test_esql_rejected_on_opensearch_backend(fake) -> None:
    # Rebind the registry with an opensearch ES target -> ES|QL must refuse.
    from opsrag.config import EnvironmentsConfig, EnvironmentTarget, EsTarget
    from opsrag.environments import bind_environments

    class _Cfg:
        environments = EnvironmentsConfig(
            default="default",
            targets={
                "default": EnvironmentTarget(
                    elasticsearch=EsTarget(
                        reach="direct", url="http://es.local:9200",
                        backend="opensearch", index_pattern="*",
                    ),
                ),
            },
        )

    bind_environments(_Cfg())
    with pytest.raises(MCPElasticsearchError) as ei:
        await get_tool("elasticsearch_esql_query").handler(None, {"query": "FROM x"})
    assert ei.value.reason == "backend"


async def test_cluster_health(fake) -> None:
    res = await fake.call("elasticsearch_cluster_health", {})
    assert res["status"] == "green"
    assert res["number_of_nodes"] == 3
