import pytest

from opsrag.config_mcp import ExternalMCPConfigBlock
from opsrag.mcp.external.connector import build_external_connector


class _FakeClient:
    async def list_tools(self):
        return [
            {"name": "find_projects", "description": "d", "inputSchema": {"type": "object", "properties": {}}},
            {"name": "search_events", "description": "d", "inputSchema": {}},
            {"name": "execute_sentry_tool", "description": "meta", "inputSchema": {}},
            {"name": "update_issue", "description": "write", "inputSchema": {}},
        ]

    async def call_tool(self, name, args):
        return {"content": [{"type": "text", "text": f"called:{name}:{args.get('q')}"}]}


@pytest.mark.asyncio
async def test_build_filters_allowlist_and_denylist_and_namespaces():
    cfg = ExternalMCPConfigBlock(
        enabled=True, restricted=False, url="https://x/mcp",
        tool_allowlist=["find_projects", "search_events", "execute_sentry_tool"],
        tool_denylist=["execute_sentry_tool", "update_issue"],
    )
    tools, kept = await build_external_connector("sentry_mcp", cfg, _FakeClient())
    names = sorted(t.name for t in tools)
    # execute_sentry_tool dropped by denylist even though allowlisted; update_issue not allowlisted.
    assert names == ["sentry_mcp_find_projects", "sentry_mcp_search_events"]
    assert set(kept) == {"find_projects", "search_events"}
    # inputSchema (camel) -> input_schema; missing schema defaults to object.
    t = next(t for t in tools if t.name == "sentry_mcp_search_events")
    assert t.input_schema == {"type": "object", "properties": {}}


@pytest.mark.asyncio
async def test_wrapped_handler_ignores_arg1_and_proxies():
    cfg = ExternalMCPConfigBlock(enabled=True, url="https://x/mcp", tool_allowlist=["find_projects"])
    tools, _ = await build_external_connector("sentry_mcp", cfg, _FakeClient())
    handler = tools[0].handler
    # Proxy path passes client=None as arg1 -> must be ignored.
    out = await handler(None, {"q": "hi"})
    assert out["content"][0]["text"] == "called:find_projects:hi"


@pytest.mark.asyncio
async def test_register_mutates_registry_and_all_tools_in_place():
    from opsrag import mcp as mcp_pkg
    from opsrag.config import Settings
    from opsrag.mcp.external.connector import register_external_connectors
    from opsrag.mcp.registry import REGISTRY
    from opsrag.mcp_server import registry_loader as rl

    before_id = id(mcp_pkg.ALL_MCP_TOOLS)
    before_len = len(mcp_pkg.ALL_MCP_TOOLS)

    s = Settings.model_validate({"external_mcp": {"sentry_mcp": {
        "enabled": True, "restricted": False, "url": "https://x/mcp",
        "tool_allowlist": ["find_projects"], "tool_denylist": [],
    }}})

    async def _factory(name, blk):
        return _FakeClient()

    names = await register_external_connectors(s, client_factory=_factory)
    try:
        assert names == ["sentry_mcp"]
        # extended in place, not rebound
        assert id(mcp_pkg.ALL_MCP_TOOLS) == before_id
        assert len(mcp_pkg.ALL_MCP_TOOLS) == before_len + 1
        assert "sentry_mcp" in REGISTRY
        assert REGISTRY["sentry_mcp"].tool_names == ("sentry_mcp_find_projects",)
        # tool->connector reverse map rebuilt
        assert rl.connector_for_tool("sentry_mcp_find_projects") == "sentry_mcp"
    finally:
        # cleanup so the test is idempotent
        REGISTRY.pop("sentry_mcp", None)
        mcp_pkg.ALL_MCP_TOOLS[:] = [t for t in mcp_pkg.ALL_MCP_TOOLS if not t.name.startswith("sentry_mcp_")]
        rl.rebuild_tool_to_connector()
        rl.set_active_enabled(None)


@pytest.mark.asyncio
async def test_second_registration_does_not_drop_earlier_external():
    """A second (widening) registration call must keep the FIRST call's
    external connector tools in the active-enabled set. Regression for the
    REPLACE-semantics bug where `set_active_enabled` was fed only the
    connectors registered THIS call, dropping earlier ones skipped by the
    `if name in REGISTRY: continue` idempotency guard."""
    from opsrag import mcp as mcp_pkg
    from opsrag.config import Settings
    from opsrag.mcp.external.connector import register_external_connectors
    from opsrag.mcp.registry import REGISTRY
    from opsrag.mcp_server import registry_loader as rl

    async def _factory(name, blk):
        return _FakeClient()

    s1 = Settings.model_validate({"external_mcp": {"sentry_mcp": {
        "enabled": True, "url": "https://x/mcp", "tool_allowlist": ["find_projects"]}}})
    s2 = Settings.model_validate({"external_mcp": {
        "sentry_mcp": {"enabled": True, "url": "https://x/mcp", "tool_allowlist": ["find_projects"]},
        "other_mcp": {"enabled": True, "url": "https://y/mcp", "tool_allowlist": ["find_projects"]}}})
    try:
        await register_external_connectors(s1, client_factory=_factory)
        await register_external_connectors(s2, client_factory=_factory)  # widening call
        active = rl.active_enabled_tool_names()
        assert active is not None
        assert "sentry_mcp_find_projects" in active   # earlier connector NOT dropped
        assert "other_mcp_find_projects" in active     # new connector present
    finally:
        for n in ("sentry_mcp", "other_mcp"):
            REGISTRY.pop(n, None)
        mcp_pkg.ALL_MCP_TOOLS[:] = [t for t in mcp_pkg.ALL_MCP_TOOLS
                                     if not (t.name.startswith("sentry_mcp_") or t.name.startswith("other_mcp_"))]
        rl.rebuild_tool_to_connector()
        rl.set_active_enabled(None)


@pytest.mark.asyncio
async def test_proxy_admits_registered_external_tools():
    from opsrag import mcp as mcp_pkg
    from opsrag.config import Settings
    from opsrag.mcp.external.connector import register_external_connectors
    from opsrag.mcp.registry import REGISTRY
    from opsrag.mcp_server import registry_loader as rl
    from opsrag.mcp_server.registry import build_external_registry

    s = Settings.model_validate({"external_mcp": {"sentry_mcp": {
        "enabled": True, "url": "https://x/mcp", "tool_allowlist": ["find_projects"]}}})

    async def _factory(name, blk):
        return _FakeClient()

    await register_external_connectors(s, client_factory=_factory)
    try:
        names = {t.name for t in build_external_registry()}
        assert "sentry_mcp_find_projects" in names
    finally:
        REGISTRY.pop("sentry_mcp", None)
        mcp_pkg.ALL_MCP_TOOLS[:] = [t for t in mcp_pkg.ALL_MCP_TOOLS if not t.name.startswith("sentry_mcp_")]
        rl.rebuild_tool_to_connector()
        rl.set_active_enabled(None)


@pytest.mark.asyncio
async def test_meta_executors_and_writes_are_dropped_even_if_allowlisted():
    """Regression guard: even if an operator accidentally allowlists the
    meta-executor/write tools, the denylist (belt-and-braces) must still drop
    them. Only find_projects should survive."""
    cfg = ExternalMCPConfigBlock(
        enabled=True, url="https://x/mcp",
        tool_allowlist=["find_projects", "execute_sentry_tool", "search_sentry_tools",
                        "update_issue", "add_issue_note"],
        tool_denylist=["execute_sentry_tool", "search_sentry_tools", "update_issue", "add_issue_note"],
    )

    class _C:
        async def list_tools(self):
            return [{"name": n, "description": "d", "inputSchema": {}} for n in
                    ["find_projects", "execute_sentry_tool", "search_sentry_tools", "update_issue", "add_issue_note"]]

        async def call_tool(self, name, args):
            return {}

    tools, kept = await build_external_connector("sentry_mcp", cfg, _C())
    assert kept == ["find_projects"]


@pytest.mark.asyncio
async def test_write_verb_backstop_drops_writes_even_with_empty_denylist():
    """M1 regression: even with an ALL-permissive allowlist and an EMPTY
    denylist, the write-verb backstop must still drop tools whose name
    contains a write verb, while the pilot's read tools survive."""
    cfg = ExternalMCPConfigBlock(
        enabled=True, url="https://x/mcp",
        tool_allowlist=[
            "find_projects", "search_events", "get_sentry_resource",
            "delete_project", "create_dashboard", "update_issue",
        ],
        tool_denylist=[],
    )

    class _C:
        async def list_tools(self):
            return [
                {"name": n, "description": "d", "inputSchema": {}}
                for n in [
                    "find_projects", "search_events", "get_sentry_resource",
                    "delete_project", "create_dashboard", "update_issue",
                ]
            ]

        async def call_tool(self, name, args):
            return {}

    tools, kept = await build_external_connector("sentry_mcp", cfg, _C())
    assert kept == ["find_projects", "search_events", "get_sentry_resource"]


# --- schema sanitization + UI-display metadata (Vertex tool-spec safety) ------

from opsrag.mcp.external.connector import _sanitize_schema  # noqa: E402


def test_sanitize_collapses_anyof_null_union():
    schema = {
        "type": "object",
        "properties": {
            "organizationSlug": {"anyOf": [{"type": "string"}, {"type": "null"}],
                                 "description": "org"},
            "limit": {"anyOf": [{"type": "null"}, {"type": "integer"}]},
        },
        "required": ["organizationSlug"],
    }
    out = _sanitize_schema(schema)
    props = out["properties"]
    assert props["organizationSlug"] == {"type": "string", "description": "org"}
    assert props["limit"] == {"type": "integer"}
    assert "anyOf" not in props["organizationSlug"] and "anyOf" not in props["limit"]


def test_sanitize_type_list_drops_null():
    assert _sanitize_schema({"type": ["string", "null"]}) == {"type": "string"}
    assert _sanitize_schema({"type": ["null", "integer"]}) == {"type": "integer"}


def test_sanitize_drops_draft_metadata_and_recurses_items():
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "definitions": {"X": {"type": "string"}},
        "type": "object",
        "properties": {
            "tags": {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
        },
    }
    out = _sanitize_schema(schema)
    assert "$schema" not in out and "definitions" not in out
    assert out["properties"]["tags"]["items"] == {"type": "string"}


@pytest.mark.asyncio
async def test_build_sanitizes_upstream_schema():
    cfg = ExternalMCPConfigBlock(enabled=True, url="https://x/mcp", tool_allowlist=["find_projects"])
    class _C:
        async def list_tools(self):
            return [{"name": "find_projects", "description": "d",
                     "inputSchema": {"type": "object", "properties": {
                         "slug": {"anyOf": [{"type": "string"}, {"type": "null"}]}}}}]
        async def call_tool(self, name, args): return {}
    tools, _ = await build_external_connector("sentry_mcp", cfg, _C())
    assert tools[0].input_schema["properties"]["slug"] == {"type": "string"}


@pytest.mark.asyncio
async def test_registry_entry_has_display_name_and_required_env():
    from opsrag import mcp as mcp_pkg
    from opsrag.config import Settings
    from opsrag.mcp.external.connector import register_external_connectors
    from opsrag.mcp.registry import REGISTRY
    from opsrag.mcp_server import registry_loader as rl

    async def _factory(name, blk):
        return _FakeClient()

    s = Settings.model_validate({"external_mcp": {"sentry_mcp": {
        "enabled": True, "url": "https://x/mcp", "display_name": "Sentry",
        "auth_env": "SENTRY_MCP_TOKEN", "tool_allowlist": ["find_projects"]}}})
    await register_external_connectors(s, client_factory=_factory)
    try:
        integ = REGISTRY["sentry_mcp"]
        assert integ.display_name == "Sentry"
        assert integ.required_env == ("SENTRY_MCP_TOKEN",)
    finally:
        REGISTRY.pop("sentry_mcp", None)
        mcp_pkg.ALL_MCP_TOOLS[:] = [t for t in mcp_pkg.ALL_MCP_TOOLS if not t.name.startswith("sentry_mcp_")]
        rl.rebuild_tool_to_connector()
        rl.set_active_enabled(None)


@pytest.mark.asyncio
async def test_external_connector_classified_external_native_builtin():
    """The /integrations endpoint tags origin via config_type subclass check."""
    from opsrag import mcp as mcp_pkg
    from opsrag.config import Settings
    from opsrag.config_mcp import ExternalMCPConfigBlock
    from opsrag.mcp.external.connector import register_external_connectors
    from opsrag.mcp.registry import REGISTRY
    from opsrag.mcp_server import registry_loader as rl

    async def _factory(name, blk):
        return _FakeClient()

    s = Settings.model_validate({"external_mcp": {"sentry_mcp": {
        "enabled": True, "url": "https://x/mcp", "tool_allowlist": ["find_projects"]}}})
    await register_external_connectors(s, client_factory=_factory)
    try:
        assert issubclass(REGISTRY["sentry_mcp"].config_type, ExternalMCPConfigBlock)  # -> external
        # a native connector is NOT an external config subclass -> builtin
        native = next(n for n in REGISTRY if n != "sentry_mcp")
        assert not issubclass(REGISTRY[native].config_type, ExternalMCPConfigBlock)
    finally:
        REGISTRY.pop("sentry_mcp", None)
        mcp_pkg.ALL_MCP_TOOLS[:] = [t for t in mcp_pkg.ALL_MCP_TOOLS if not t.name.startswith("sentry_mcp_")]
        rl.rebuild_tool_to_connector()
        rl.set_active_enabled(None)


def test_native_sentry_tools_exposed_on_proxy():
    """Native sentry read tools must be in the proxy allowlist so the
    opsrag-mcp-proxy surfaces them when the connector is enabled. (They were
    historically absent because the connector shipped disabled.)"""
    from opsrag.mcp import SENTRY_TOOLS
    from opsrag.mcp_server.registry import SAFE_FOR_EXTERNAL_TOOLS, build_external_registry

    sentry_names = {t.name for t in SENTRY_TOOLS}
    assert sentry_names <= SAFE_FOR_EXTERNAL_TOOLS  # all admitted, none missing
    exposed = {t.name for t in build_external_registry() if t.name.startswith("sentry_")}
    assert exposed == sentry_names
