"""LiteLLM LLM provider -- routes chat completions through LiteLLM so any
LiteLLM-supported model (Gemini, OpenAI, Anthropic, Bedrock, or a self-hosted
OpenAI-compatible / vLLM / TGI endpoint via ``api_base``) becomes a config
flip behind the ``LLMProvider`` protocol.

LiteLLM provider-string convention: ``model`` is passed verbatim to
``litellm.acompletion`` using LiteLLM's ``<provider>/<model>`` convention,
e.g. ``gemini/gemini-2.5-flash``, ``openai/gpt-4o``,
``anthropic/claude-sonnet-4-20250514``, ``bedrock/anthropic.claude-...``, or
``openai/<model>`` against a self-hosted server when paired with ``api_base``.

Structured output mirrors :mod:`opsrag.llms.openai`: an in-prompt JSON
instruction plus a parse + ``model_validate`` (LiteLLM's structured-output
support varies by backend, so the in-prompt approach is the portable one).

``litellm`` is imported lazily inside :meth:`generate` so this module imports
cleanly without the optional dependency installed.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from pydantic import BaseModel

from opsrag.interfaces.llm import LLMResponse
from opsrag.llms.content import to_openai_content


class LiteLLMLLM:
    def __init__(
        self,
        model: str,
        default_max_tokens: int = 4096,
        api_base: str | None = None,
        api_key_env: str | None = None,
    ):
        self._model = model
        self._default_max_tokens = default_max_tokens
        self._api_base = api_base
        self._api_key = os.environ.get(api_key_env) if api_key_env else None

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
        # response_schema (raw JSON schema) is not wired to a native
        # structured-output mode here; validated structured output goes
        # through generate_structured(). response_format is passed through
        # so callers can opt into {"type": "json_object"} on backends that
        # support it.
        _ = response_schema
        import litellm

        start = time.perf_counter()

        # LiteLLM follows the OpenAI wire format: the system prompt is a
        # leading system message rather than a top-level field.
        full_messages: list[dict] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(
            {"role": m["role"], "content": to_openai_content(m.get("content", ""))}
            for m in messages
        )

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": full_messages,
            "max_tokens": max_tokens or self._default_max_tokens,
            "temperature": temperature,
        }
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if response_format:
            kwargs["response_format"] = response_format

        resp = await litellm.acompletion(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        content = resp.choices[0].message.content or ""
        model = getattr(resp, "model", None) or self._model

        # LiteLLM normalises usage to OpenAI's shape when the backend reports
        # it; estimate as a fallback so cost telemetry still populates.
        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
        out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
        if not in_tok:
            from opsrag.tokenization import estimate_tokens
            in_tok = sum(
                estimate_tokens(str(m.get("content", ""))) for m in full_messages
            )
        if not out_tok:
            from opsrag.tokenization import estimate_tokens
            out_tok = estimate_tokens(content)

        from opsrag.usage import tracker
        tracker.record(
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            purpose=purpose,
        )

        return LLMResponse(
            content=content,
            model=model,
            usage={"input_tokens": in_tok, "output_tokens": out_tok},
            latency_ms=latency_ms,
            raw_response=resp,
        )

    @staticmethod
    def _to_openai_messages(messages: list[dict], system_prompt: str | None) -> list[dict]:
        """Translate OpsRAG's internal tool-loop message format into the
        OpenAI wire format litellm expects.

        Internal roles: ``user``/``assistant`` (plain text), ``tool_call``
        (``{name, args}`` -- a prior assistant function call), ``tool_result``
        (``{name, response}`` -- a function's output). OpenAI needs the assistant
        ``tool_calls`` block (each with an ``id``) followed by ``role:"tool"``
        messages whose ``tool_call_id`` matches -- so we synthesize ids and pair
        results to calls FIFO-by-name.
        """
        out: list[dict] = []
        if system_prompt:
            out.append({"role": "system", "content": system_prompt})
        pending_ids: dict[str, list[str]] = {}
        counter = 0
        i, n = 0, len(messages)
        while i < n:
            m = messages[i]
            role = m.get("role", "user")
            if role in ("user", "assistant"):
                out.append({"role": role, "content": m.get("content", "") or ""})
                i += 1
            elif role == "tool_call":
                calls = []
                while i < n and messages[i].get("role") == "tool_call":
                    tc = messages[i]
                    cid = f"call_{counter}"
                    counter += 1
                    calls.append({
                        "id": cid, "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("args", {}) or {}),
                        },
                    })
                    pending_ids.setdefault(tc["name"], []).append(cid)
                    i += 1
                out.append({"role": "assistant", "content": None, "tool_calls": calls})
            elif role == "tool_result":
                name = m["name"]
                ids = pending_ids.get(name) or []
                cid = ids.pop(0) if ids else f"call_{counter}"
                if not ids and not pending_ids.get(name):
                    counter += 1
                content = json.dumps(m.get("response", {}) or {})
                out.append({
                    "role": "tool", "tool_call_id": cid, "name": name,
                    "content": content[:8000],
                })
                i += 1
            else:
                i += 1
        return out

    async def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        purpose: str | None = None,
    ):
        """Tool/function calling via litellm (OpenAI tool format, normalized
        across backends -- works for Qwen MaaS, Gemini, Claude, OpenAI, etc.).

        ``tools`` are MCP-shaped dicts ``{name, description, input_schema}``.
        Returns a ``ToolCallingResponse`` (same dataclass the Vertex provider
        uses) with either ``tool_calls`` or final ``text``.
        """
        # Reuse the shared dataclasses; lazy import keeps module import light.
        import litellm

        from opsrag.llms.vertex import ToolCall, ToolCallingResponse

        start = time.perf_counter()
        oai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
            for t in tools
        ]
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": self._to_openai_messages(messages, system_prompt),
            "tools": oai_tools,
            "tool_choice": "auto",
            "max_tokens": max_tokens or self._default_max_tokens,
            "temperature": temperature,
        }
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if self._api_key:
            kwargs["api_key"] = self._api_key

        resp = await litellm.acompletion(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        msg = resp.choices[0].message
        text = msg.content or ""
        tool_calls: list = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            raw = getattr(fn, "arguments", "") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append(ToolCall(name=getattr(fn, "name", "") or "", args=args))

        model = getattr(resp, "model", None) or self._model
        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
        out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
        from opsrag.usage import tracker as _t
        _t.record(
            model=model, input_tokens=in_tok, output_tokens=out_tok,
            latency_ms=latency_ms, purpose=purpose or "tool_calling",
        )
        return ToolCallingResponse(
            tool_calls=tool_calls,
            text=text,
            model=model,
            usage={"input_tokens": in_tok, "output_tokens": out_tok},
            latency_ms=latency_ms,
            raw_response=resp,
        )

    async def generate_structured(
        self,
        messages: list[dict],
        schema: type,
        system_prompt: str | None = None,
        purpose: str | None = None,
        *,
        max_tokens: int | None = None,
    ) -> Any:
        """Force JSON output matching a Pydantic v2 model via an in-prompt
        schema instruction, then parse + validate (portable across LiteLLM
        backends)."""
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise TypeError("schema must be a pydantic.BaseModel subclass")

        json_schema = schema.model_json_schema()
        instruction = (
            "Respond ONLY with a single JSON object that matches this schema. "
            "No prose, no code fences.\n\n"
            f"Schema:\n{json.dumps(json_schema, indent=2)}"
        )
        system = f"{system_prompt}\n\n{instruction}" if system_prompt else instruction

        # LiteLLM may proxy Gemini, whose thinking tokens count against
        # max_output_tokens unless response_schema is set; this in-prompt
        # path does NOT set one, so a small structured-gate cap (e.g. the
        # gates' 128) would truncate thinking -> empty output -> json parse
        # fails. Ignore a small cap here to keep gate verdicts unchanged:
        # floor it at the provider default so it can never cap below the safe
        # value. (Net effect identical to dropping a small cap; left None it
        # preserves generate's default behavior so existing callers are
        # unchanged.)
        gen_kwargs: dict[str, Any] = {
            "messages": messages,
            "system_prompt": system,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "purpose": purpose,
        }
        if max_tokens is not None:
            gen_kwargs["max_tokens"] = max(max_tokens, self._default_max_tokens)
        resp = await self.generate(**gen_kwargs)

        text = resp.content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        data = json.loads(text)
        return schema.model_validate(data)
