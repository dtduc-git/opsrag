"""`_mcp_schema_to_vertex` must normalize JSON-Schema union types that
remote MCP servers emit for optional params -- Vertex's FunctionDeclaration
has no NULL type, so any "null" leaking through kills the WHOLE tool call
at request build.

Observed in prod 2026-07-13 (00:41 UTC), minutes after the sentry_mcp
external connector was admitted: every pro-tier triage on the native
VertexAILLM died in ~23ms with
  "Failed to parse type field: Invalid enum value NULL for enum type
   google.cloud.aiplatform.v1beta1.Type at Schema.properties[environment].type"
and the query silently fell through to retrieval -- the tool lane never ran.
Root cause: Sentry MCP declares optional params as
`anyOf: [{"type": "null"}, {"type": "string"}]` and the flattener picked
the FIRST typed branch (null). `type: ["string", "null"]` lists and bare
`type: "null"` are the same class of input.
"""
from __future__ import annotations

import pytest

pytest.importorskip("vertexai")  # optional [vertex] extra; skip when absent

from opsrag.llms.vertex import _mcp_schema_to_vertex


def _prop_type(schema: dict, name: str = "environment"):
    return _mcp_schema_to_vertex(schema)["properties"][name].get("type")


def test_anyof_null_first_picks_non_null_branch():
    schema = {"type": "object", "properties": {"environment": {
        "anyOf": [{"type": "null"}, {"type": "string"}],
    }}}
    assert _prop_type(schema) == "string"


def test_anyof_all_null_falls_back_to_string():
    schema = {"type": "object", "properties": {"environment": {
        "anyOf": [{"type": "null"}],
    }}}
    assert _prop_type(schema) == "string"


def test_type_list_picks_first_non_null():
    schema = {"type": "object", "properties": {"environment": {
        "type": ["null", "integer"],
    }}}
    assert _prop_type(schema) == "integer"


def test_type_list_only_null_falls_back_to_string():
    schema = {"type": "object", "properties": {"environment": {
        "type": ["null"],
    }}}
    assert _prop_type(schema) == "string"


def test_bare_null_type_falls_back_to_string():
    schema = {"type": "object", "properties": {"environment": {"type": "null"}}}
    assert _prop_type(schema) == "string"


def test_nested_union_inside_array_items_normalized():
    schema = {"type": "object", "properties": {"tags": {
        "type": "array",
        "items": {"anyOf": [{"type": "null"}, {"type": "string"}]},
    }}}
    cleaned = _mcp_schema_to_vertex(schema)
    assert cleaned["properties"]["tags"]["items"]["type"] == "string"


def test_every_normalized_shape_is_accepted_by_functiondeclaration():
    # End check against the real SDK parser -- the thing prod actually hit.
    from vertexai.generative_models import FunctionDeclaration

    schema = {"type": "object", "properties": {
        "environment": {"anyOf": [{"type": "null"}, {"type": "string"}],
                        "description": "env filter"},
        "limit": {"type": ["null", "integer"]},
        "query": {"type": "string"},
        "tags": {"type": "array",
                 "items": {"anyOf": [{"type": "null"}, {"type": "string"}]}},
    }, "required": ["query"]}
    FunctionDeclaration(
        name="search_events", description="d",
        parameters=_mcp_schema_to_vertex(schema),
    )  # must not raise


def test_plain_schema_unchanged():
    schema = {"type": "object", "properties": {
        "repo": {"type": "string", "description": "Repo name."},
    }, "required": ["repo"]}
    cleaned = _mcp_schema_to_vertex(schema)
    assert cleaned["properties"]["repo"] == {"type": "string", "description": "Repo name."}
    assert cleaned["required"] == ["repo"]
