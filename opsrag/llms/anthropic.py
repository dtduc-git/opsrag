"""Anthropic LLM provider -- direct SDK calls, no LangChain wrappers."""
from __future__ import annotations

import json
import time
from typing import Any

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from opsrag.interfaces.llm import LLMResponse
from opsrag.llms.content import to_anthropic_content
from opsrag.llms.json_extract import extract_first_json_object


class AnthropicLLM:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        default_max_tokens: int = 4096,
        *,
        timeout: float | None = None,
        max_retries: int | None = None,
    ):
        self._api_key = api_key
        self._client: AsyncAnthropic | None = None
        self._model = model
        self._default_max_tokens = default_max_tokens
        # Optional client-level robustness knobs. Left None, the AsyncAnthropic
        # SDK keeps its own native defaults (10-min timeout, 2 retries) -- so
        # existing callers see identical behavior.
        self._timeout = timeout
        self._max_retries = max_retries

    def _get_client(self) -> AsyncAnthropic:
        if self._client is None:
            kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._timeout is not None:
                kwargs["timeout"] = self._timeout
            if self._max_retries is not None:
                kwargs["max_retries"] = self._max_retries
            self._client = AsyncAnthropic(**kwargs)
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
        # response_schema is a no-op for Anthropic direct -- the Anthropic
        # SDK doesn't expose a JSON-schema response mode; callers that
        # need structured output should use generate_structured(), which
        # appends a schema instruction to the system prompt.
        _ = response_schema
        start = time.perf_counter()
        converted = [
            {"role": m["role"], "content": to_anthropic_content(m.get("content", ""))}
            for m in messages
        ]
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": converted,
            "max_tokens": max_tokens or self._default_max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        resp = await self._get_client().messages.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        content = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )

        in_tok = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens

        from opsrag.usage import tracker
        tracker.record(
            model=resp.model,
            input_tokens=in_tok, output_tokens=out_tok, latency_ms=latency_ms,
            purpose=purpose,
        )

        return LLMResponse(
            content=content,
            model=resp.model,
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
        """Force JSON output that matches a Pydantic v2 model.

        Uses an appended instruction -- Anthropic doesn't expose native JSON mode,
        so we enforce the schema in-prompt and parse/validate on return.

        ``max_tokens`` caps the output. Boolean/verdict callers pass a small cap
        (e.g. 128) safely above the tiny payload; left None it preserves
        ``generate``'s default-4096 behavior, so existing callers are unchanged.
        """
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise TypeError("schema must be a pydantic.BaseModel subclass")

        json_schema = schema.model_json_schema()
        instruction = (
            "Respond ONLY with a single JSON object that matches this schema. "
            "No prose, no code fences.\n\n"
            f"Schema:\n{json.dumps(json_schema, indent=2)}"
        )
        system = f"{system_prompt}\n\n{instruction}" if system_prompt else instruction

        gen_kwargs: dict[str, Any] = {
            "messages": messages,
            "system_prompt": system,
            "temperature": 0.0,
            "purpose": purpose,
        }
        if max_tokens is not None:
            gen_kwargs["max_tokens"] = max_tokens
        resp = await self.generate(**gen_kwargs)

        data = extract_first_json_object(resp.content or "")
        return schema.model_validate(data)
