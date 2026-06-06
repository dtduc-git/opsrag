"""Vertex AI LLM provider -- supports Claude on Vertex and Gemini.

For Claude on Vertex:
  model = "claude-sonnet-4@20250514"  (Anthropic models via Vertex)

For Gemini:
  model = "gemini-2.0-flash"

Auth: Uses Application Default Credentials (ADC).
  - Local: gcloud auth application-default login
  - GKE: Workload Identity
  - CI: GOOGLE_APPLICATION_CREDENTIALS service account JSON

Requires: pip install google-cloud-aiplatform anthropic[vertex]
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from opsrag.interfaces.llm import LLMResponse

_log = logging.getLogger("opsrag.llms.vertex")


@dataclass
class VertexResult:
    """M2 -- token-attribution-friendly summary of one Vertex LLM call.

    Lighter than ``LLMResponse`` and intentionally pure-data: the
    ``on_usage`` hook receives one of these per generation and forwards
    the numbers (plus the current request's ``user_oid`` from a
    contextvar set by the integration layer) to
    ``UsagePersistence.enqueue``.

    `text` is a convenience alias for `content`. The spec mentions
    callers using `result.text`; keeping both names lets either style
    work without ambiguity.
    """

    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    latency_ms: float = 0.0

    # Convenience: a number of code paths in this codebase historically
    # used `.content`. Aliased so a future refactor can rename freely.
    @property
    def content(self) -> str:
        return self.text


# Type alias for the per-call usage hook. May be sync or async; we
# `inspect.iscoroutine` the return value and await if needed.
OnUsageHook = Callable[[VertexResult], Any | Awaitable[None]]


@dataclass
class ToolCall:
    """Phase 03 Pillar 2 -- a function call the LLM emitted during
    `generate_with_tools()`. Args are already a plain dict."""
    name: str
    args: dict


@dataclass
class ToolCallingResponse:
    """Phase 03 Pillar 2 -- tool-calling result. Either `tool_calls` is
    non-empty (LLM wants to invoke functions) OR `text` is set (LLM is
    done and returned final text), never both meaningfully."""
    tool_calls: list[ToolCall]
    text: str
    model: str
    usage: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    raw_response: Any = None


def _mcp_schema_to_vertex(schema: dict) -> dict:
    """Translate an MCP-shape JSON Schema (`{type, properties, required, ...}`)
    to the subset Vertex's FunctionDeclaration accepts. The two are
    near-identical; we strip fields Vertex doesn't recognize and
    flatten `oneOf`/`anyOf` to `string` (Vertex ignores complex unions
    silently otherwise, which produces opaque function-calling errors).
    """
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    cleaned: dict = {}
    type_ = schema.get("type", "object")
    cleaned["type"] = type_
    if "description" in schema:
        cleaned["description"] = schema["description"]
    if type_ == "object":
        props_in = schema.get("properties", {}) or {}
        props_out: dict = {}
        for name, prop in props_in.items():
            if not isinstance(prop, dict):
                continue
            if "oneOf" in prop or "anyOf" in prop:
                # Vertex doesn't accept oneOf/anyOf in function params.
                # Pick the first concrete-typed branch, fall back to string.
                branches = prop.get("oneOf") or prop.get("anyOf") or []
                picked = next(
                    (b for b in branches if isinstance(b, dict) and "type" in b),
                    {"type": "string"},
                )
                merged = {k: v for k, v in prop.items() if k not in ("oneOf", "anyOf")}
                merged.update({k: v for k, v in picked.items() if k not in merged})
                props_out[name] = _mcp_schema_to_vertex(merged)
            else:
                props_out[name] = _mcp_schema_to_vertex(prop)
        cleaned["properties"] = props_out
        if "required" in schema:
            cleaned["required"] = list(schema["required"])
    elif type_ == "array":
        items = schema.get("items")
        if isinstance(items, dict):
            cleaned["items"] = _mcp_schema_to_vertex(items)
    if "enum" in schema:
        cleaned["enum"] = list(schema["enum"])
    return cleaned


def _messages_to_gemini_contents(messages: list[dict]) -> list:
    """Translate role/content + tool-call/tool-result messages to Vertex
    `Content`/`Part` objects. Roles supported:
      - user / assistant -- plain text
      - tool_call -- assistant's prior function call (name, args)
      - tool_result -- function execution result (name, response)
    """
    from vertexai.generative_models import Content, Part

    contents: list = []
    for msg in messages:
        role = msg.get("role", "user")
        if role in ("user", "assistant"):
            text = msg.get("content", "") or ""
            mapped_role = "user" if role == "user" else "model"
            contents.append(Content(role=mapped_role, parts=[Part.from_text(text)]))
        elif role == "tool_call":
            # Re-emit a model turn that's an assistant function-call --
            # Vertex needs this to keep the conversation aligned with the
            # function_response that follows.
            name = msg["name"]
            args = msg.get("args", {}) or {}
            from vertexai.generative_models import Part as _P
            contents.append(
                Content(role="model", parts=[_P.from_dict({
                    "function_call": {"name": name, "args": args},
                })])
            )
        elif role == "tool_result":
            name = msg["name"]
            response = msg.get("response", {}) or {}
            contents.append(
                Content(role="user", parts=[Part.from_function_response(
                    name=name, response={"result": response},
                )])
            )
    return contents


def _extract_gemini_text(resp: Any) -> str:
    # `resp.text` raises "Multiple content parts are not supported" when the
    # candidate has more than one part (e.g. text + thought, or text + function
    # call). Walk parts ourselves and concat any with a `text` attribute.
    chunks: list[str] = []
    for cand in getattr(resp, "candidates", None) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                chunks.append(text)
    return "".join(chunks)


class VertexAILLM:
    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        project: str | None = None,
        location: str | None = None,
        default_max_tokens: int = 4096,
        on_usage: OnUsageHook | None = None,
    ):
        import os
        self._model = model
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location or os.environ.get("CLOUD_ML_REGION", "us-central1")
        self._default_max_tokens = default_max_tokens
        self._is_claude = "claude" in model.lower()
        self._client: Any = None
        # M2 -- optional usage callback. Wired by the integration layer
        # (factory / lifespan) to UsagePersistence.enqueue plus the
        # current-request user_oid pulled from a contextvar. Failures
        # are swallowed by `_fire_on_usage` so telemetry never breaks
        # the path that produced it.
        self.on_usage: OnUsageHook | None = on_usage

    def set_on_usage(self, hook: OnUsageHook | None) -> None:
        """Allow the integration layer to attach/detach the usage hook
        post-construction (e.g. after `UsagePersistence` is open)."""
        self.on_usage = hook

    async def _fire_on_usage(
        self,
        *,
        text: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        latency_ms: float,
    ) -> None:
        """Build a VertexResult and dispatch it to ``self.on_usage`` if set.

        Swallows exceptions -- usage telemetry must never break the
        request path. Supports both sync and async hooks so the
        integration layer can pick whichever is easier to wire.
        """
        if self.on_usage is None:
            return
        try:
            result = VertexResult(
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model=model,
                latency_ms=latency_ms,
            )
            ret = self.on_usage(result)
            if inspect.isawaitable(ret):
                await ret
        except Exception as exc:  # noqa: BLE001 -- never re-raise
            _log.debug("on_usage hook failed: %s", exc)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if self._is_claude:
            from anthropic import AnthropicVertex
            self._client = AnthropicVertex(
                project_id=self._project,
                region=self._location,
            )
        else:
            import vertexai
            from vertexai.generative_models import GenerativeModel
            vertexai.init(project=self._project, location=self._location)
            self._client = GenerativeModel(self._model)

        return self._client

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: dict | None = None,
        purpose: str | None = None,
        response_schema: dict | None = None,
    ) -> LLMResponse:
        start = time.perf_counter()

        if self._is_claude:
            # Anthropic-on-Vertex doesn't expose response_schema; the
            # caller's prompt-only JSON instructions remain authoritative.
            resp = await self._generate_claude(
                messages, system_prompt, temperature, max_tokens, start
            )
        else:
            resp = await self._generate_gemini(
                messages, system_prompt, temperature, max_tokens, start,
                response_schema=response_schema,
            )
        # Single point of usage recording for every Vertex LLM call.
        # `purpose` lets the dashboard split query-side vs indexing-side.
        from opsrag.usage import tracker as _t
        _t.record(
            model=resp.model,
            input_tokens=resp.usage.get("input_tokens", 0),
            output_tokens=resp.usage.get("output_tokens", 0),
            latency_ms=resp.latency_ms,
            purpose=purpose,
        )
        # M2 -- fire the per-call hook so the integration layer can
        # attribute this call to the current request's user_oid.
        await self._fire_on_usage(
            text=resp.content,
            prompt_tokens=resp.usage.get("input_tokens", 0),
            completion_tokens=resp.usage.get("output_tokens", 0),
            model=resp.model,
            latency_ms=resp.latency_ms,
        )
        return resp

    async def _generate_claude(
        self, messages, system_prompt, temperature, max_tokens, start
    ) -> LLMResponse:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        # Anthropic Vertex SDK is sync -- without to_thread, the LLM
        # round-trip (1-5s) blocks the event loop and wedges /health.
        resp = await asyncio.to_thread(client.messages.create, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        content = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        return LLMResponse(
            content=content,
            model=resp.model,
            usage={
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
            latency_ms=latency_ms,
            raw_response=resp,
        )

    async def _generate_gemini(
        self, messages, system_prompt, temperature, max_tokens, start,
        response_schema: dict | None = None,
    ) -> LLMResponse:
        from vertexai.generative_models import GenerationConfig

        client = self._get_client()
        prompt_parts = []
        if system_prompt:
            prompt_parts.append(system_prompt)
        for msg in messages:
            prompt_parts.append(f"{msg['role']}: {msg['content']}")

        # T1.1: when response_schema is provided, force structured JSON
        # output. This makes Gemini's thinking phase a separate budget
        # from the visible-output budget, so max_tokens no longer gets
        # silently eaten by thought tokens and truncated JSON becomes
        # categorically impossible.
        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema
        config = GenerationConfig(**config_kwargs)
        # Vertex GenerativeModel.generate_content is sync -- must run in a
        # thread to keep the event loop responsive during indexing
        # (contextual chunking issues many concurrent calls).
        resp = await asyncio.to_thread(
            client.generate_content,
            "\n\n".join(prompt_parts),
            generation_config=config,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        return LLMResponse(
            content=_extract_gemini_text(resp),
            model=self._model,
            usage={
                "input_tokens": getattr(resp.usage_metadata, "prompt_token_count", 0),
                "output_tokens": getattr(resp.usage_metadata, "candidates_token_count", 0),
            },
            latency_ms=latency_ms,
            raw_response=resp,
        )

    async def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        purpose: str | None = None,
    ) -> ToolCallingResponse:
        """Phase 03 Pillar 2 -- Vertex Gemini function-calling.

        `tools` is a list of MCP-shaped tool dicts:
            {"name": str, "description": str, "input_schema": JSONSchema}

        Returns a `ToolCallingResponse` with either `tool_calls` (LLM
        wants to invoke functions) or `text` (LLM is done). Messages
        carry conversation turns including prior tool-call results -- see
        `_messages_to_gemini_contents` for the encoding.

        Claude-on-Vertex path is not implemented in Pillar 2 (would
        require Anthropic tool-use API translation); call `generate()`
        for prose-only Claude responses instead.
        """
        if self._is_claude:
            raise NotImplementedError(
                "generate_with_tools is Gemini-only in Pillar 2; "
                "Claude tool-use translation deferred"
            )

        from vertexai.generative_models import (
            FunctionDeclaration,
            GenerationConfig,
            GenerativeModel,
            Tool,
        )

        start = time.perf_counter()

        # Translate MCP-shape tool specs -> Vertex FunctionDeclarations.
        decls = [
            FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=_mcp_schema_to_vertex(t["input_schema"]),
            )
            for t in tools
        ]
        vertex_tool = Tool(function_declarations=decls)

        # Vertex needs the GenerativeModel to know about tools at
        # construction; we build a per-call instance to keep this
        # decoupled from the cached `self._client` (which has no tools).
        import vertexai
        vertexai.init(project=self._project, location=self._location)
        model = GenerativeModel(
            self._model,
            system_instruction=system_prompt,
            tools=[vertex_tool],
        )

        contents = _messages_to_gemini_contents(messages)
        config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        resp = await asyncio.to_thread(
            model.generate_content,
            contents,
            generation_config=config,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        tool_calls: list[ToolCall] = []
        text_chunks: list[str] = []
        for cand in getattr(resp, "candidates", None) or []:
            content = getattr(cand, "content", None)
            for part in getattr(content, "parts", None) or []:
                fc = getattr(part, "function_call", None)
                if fc is not None and getattr(fc, "name", None):
                    # `fc.args` is a Struct/MapComposite; coerce to plain dict.
                    args = dict(fc.args) if fc.args else {}
                    tool_calls.append(ToolCall(name=fc.name, args=args))
                    continue
                text = getattr(part, "text", None)
                if text:
                    text_chunks.append(text)

        usage = {
            "input_tokens": getattr(resp.usage_metadata, "prompt_token_count", 0),
            "output_tokens": getattr(resp.usage_metadata, "candidates_token_count", 0),
        }
        from opsrag.usage import tracker as _t
        _t.record(
            model=self._model,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            latency_ms=latency_ms,
            purpose=purpose or "tool_calling",
        )
        # M2 -- fire the hook with the aggregated tool-call text so the
        # integration layer can attribute the cost to the current user.
        await self._fire_on_usage(
            text="".join(text_chunks),
            prompt_tokens=usage["input_tokens"],
            completion_tokens=usage["output_tokens"],
            model=self._model,
            latency_ms=latency_ms,
        )

        return ToolCallingResponse(
            tool_calls=tool_calls,
            text="".join(text_chunks),
            model=self._model,
            usage=usage,
            latency_ms=latency_ms,
            raw_response=resp,
        )

    async def generate_with_tools_stream(
        self,
        messages: list[dict],
        tools: list[dict],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        purpose: str | None = None,
    ):
        """Streaming variant of `generate_with_tools` -- yields incremental
        text deltas as the model generates them, then a final dict with
        the aggregated tool_calls + full text.

        Yielded shapes:
            {"type": "text_delta", "text": "..."}         # per chunk
            {"type": "done", "response": ToolCallingResponse}

        Used by `reasoner_node` to feed live "thinking" tokens to the
        chat UI (UX inspired by Claude's reasoning indicator and o1's
        visible chain of thought).
        """
        if self._is_claude:
            raise NotImplementedError(
                "generate_with_tools_stream is Gemini-only"
            )
        from vertexai.generative_models import (
            FunctionDeclaration,
            GenerationConfig,
            GenerativeModel,
            Tool,
        )

        start = time.perf_counter()
        decls = [
            FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=_mcp_schema_to_vertex(t["input_schema"]),
            )
            for t in tools
        ]
        vertex_tool = Tool(function_declarations=decls)
        import vertexai
        vertexai.init(project=self._project, location=self._location)
        model = GenerativeModel(
            self._model,
            system_instruction=system_prompt,
            tools=[vertex_tool],
        )
        contents = _messages_to_gemini_contents(messages)
        config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # Vertex `generate_content(stream=True)` returns a sync iterator;
        # we drain it on a worker thread and bridge to async via a Queue.
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        _SENTINEL = object()

        def _drain():
            try:
                stream = model.generate_content(
                    contents, generation_config=config, stream=True,
                )
                for chunk in stream:
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

        # Fire-and-collect: drain on a thread, consume chunks on the loop.
        drain_task = loop.run_in_executor(None, _drain)

        tool_calls: list[ToolCall] = []
        text_chunks: list[str] = []
        usage_meta: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        last_chunk = None

        while True:
            item = await queue.get()
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            last_chunk = item
            # Aggregate usage as it arrives (Vertex emits cumulative
            # usage on the last chunk; we just keep overwriting).
            um = getattr(item, "usage_metadata", None)
            if um is not None:
                usage_meta = {
                    "input_tokens": getattr(um, "prompt_token_count", 0) or 0,
                    "output_tokens": getattr(um, "candidates_token_count", 0) or 0,
                }
            # Walk the chunk's parts; emit text deltas; capture tool calls.
            for cand in getattr(item, "candidates", None) or []:
                content = getattr(cand, "content", None)
                for part in getattr(content, "parts", None) or []:
                    fc = getattr(part, "function_call", None)
                    if fc is not None and getattr(fc, "name", None):
                        args = dict(fc.args) if fc.args else {}
                        tool_calls.append(ToolCall(name=fc.name, args=args))
                        continue
                    text = getattr(part, "text", None)
                    if text:
                        text_chunks.append(text)
                        yield {"type": "text_delta", "text": text}

        # Make sure the background drain task fully cleaned up.
        try:
            await drain_task
        except Exception:
            pass

        latency_ms = (time.perf_counter() - start) * 1000
        from opsrag.usage import tracker as _t
        _t.record(
            model=self._model,
            input_tokens=usage_meta["input_tokens"],
            output_tokens=usage_meta["output_tokens"],
            latency_ms=latency_ms,
            purpose=purpose or "tool_calling_stream",
        )
        # M2 -- same hook on the streaming path.
        await self._fire_on_usage(
            text="".join(text_chunks),
            prompt_tokens=usage_meta["input_tokens"],
            completion_tokens=usage_meta["output_tokens"],
            model=self._model,
            latency_ms=latency_ms,
        )
        yield {
            "type": "done",
            "response": ToolCallingResponse(
                tool_calls=tool_calls,
                text="".join(text_chunks),
                model=self._model,
                usage=usage_meta,
                latency_ms=latency_ms,
                raw_response=last_chunk,
            ),
        }

    async def generate_structured(
        self,
        messages: list[dict],
        schema: type,
        system_prompt: str | None = None,
        purpose: str | None = None,
    ) -> Any:
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise TypeError("schema must be a pydantic.BaseModel subclass")

        json_schema = schema.model_json_schema()
        instruction = (
            "Respond ONLY with a single JSON object that matches this schema. "
            "No prose, no code fences.\n\n"
            f"Schema:\n{json.dumps(json_schema, indent=2)}"
        )
        system = f"{system_prompt}\n\n{instruction}" if system_prompt else instruction

        resp = await self.generate(
            messages=messages, system_prompt=system, temperature=0.0, purpose=purpose,
        )
        text = resp.content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        data = json.loads(text)
        return schema.model_validate(data)
