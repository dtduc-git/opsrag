"""Query routing node -- classifies the query and decides retrieval strategy."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from opsrag.agent.prompts import ROUTER_SYSTEM
from opsrag.interfaces.llm import LLMProvider
from opsrag.interfaces.observability import ObservabilityProvider

# Output token cap for the route-decision gate. The structured payload is the
# small ``_RouteDecision`` object ({query_type, requires_graph, confidence}) --
# ~10-20 tokens including JSON punctuation -- so a 128-token ceiling lets the tiny
# verdict schedule faster WITHOUT any chance of truncating a valid decision.
# Quality-neutral: the cap is HONORED on non-thinking providers
# (Anthropic/Bedrock-Claude/OpenAI), while Vertex and LiteLLM-Gemini IGNORE it
# (floor it at their default) to avoid truncating Gemini thinking tokens -- which
# count against max_output_tokens unless a response_schema is set, and the
# in-prompt-schema structured path sets none. Were the cap honored on Gemini,
# truncated thinking would yield empty output -> a json parse failure -> the node
# silently falls back to "general"/no-graph (graph routing off). So the routing
# choice is identical on every provider, with or without the cap.
_GATE_MAX_TOKENS = 128

QueryTypeValue = Literal[
    "incident",
    "howto",
    "architecture",
    "config_lookup",
    "postmortem_search",
    "blast_radius",
    "dependency_map",
    "general",
]


class _RouteDecision(BaseModel):
    query_type: QueryTypeValue = Field(description="Category of the operational question")
    requires_graph: bool = Field(description="Whether knowledge graph traversal is needed")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence 0-1")


def route_query_node(llm: LLMProvider, observability: ObservabilityProvider):
    async def _route(state: dict) -> dict:
        query = state["query"]
        try:
            decision = await llm.generate_structured(
                purpose="route",
                messages=[{"role": "user", "content": query}],
                schema=_RouteDecision,
                system_prompt=ROUTER_SYSTEM,
                max_tokens=_GATE_MAX_TOKENS,
            )
            qt = decision.query_type
            req_graph = decision.requires_graph or qt in ("blast_radius", "dependency_map")
            confidence = decision.confidence
        except Exception:
            qt = "general"
            req_graph = False
            confidence = 0.0

        return {
            "query_type": qt,
            "intent_confidence": confidence,
            "requires_graph": req_graph,
            "current_step": "routed",
        }

    return _route
