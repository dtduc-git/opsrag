"""OpenAI LLM provider -- direct SDK calls, no LangChain wrappers.

Mirrors :class:`opsrag.llms.anthropic.AnthropicLLM` so the factory can swap
providers behind the ``LLMProvider`` protocol. Unlike Anthropic, OpenAI
exposes a native JSON-object response mode, which ``generate_structured``
uses for tighter schema adherence.
"""
from __future__ import annotations

import json
import time
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from opsrag.interfaces.llm import LLMResponse


class OpenAILLM:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        default_max_tokens: int = 4096,
        base_url: str | None = None,
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._client: AsyncOpenAI | None = None
        self._model = model
        self._default_max_tokens = default_max_tokens

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = AsyncOpenAI(**kwargs)
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
        # response_schema (raw JSON schema) is not wired to OpenAI's
        # json_schema mode here; callers needing validated structured output
        # use generate_structured(). response_format is passed through as-is
        # so callers can opt into {"type": "json_object"} when desired.
        _ = response_schema
        start = time.perf_counter()

        # OpenAI carries the system prompt as a leading system message rather
        # than a top-level field.
        full_messages: list[dict] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": full_messages,
            "max_tokens": max_tokens or self._default_max_tokens,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        resp = await self._get_client().chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        content = resp.choices[0].message.content or ""

        usage = resp.usage
        in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
        out_tok = getattr(usage, "completion_tokens", 0) if usage else 0

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
    ) -> Any:
        """Force JSON output that matches a Pydantic v2 model, using OpenAI's
        native ``json_object`` response mode plus an in-prompt schema."""
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
            messages=messages,
            system_prompt=system,
            temperature=0.0,
            response_format={"type": "json_object"},
            purpose=purpose,
        )

        text = resp.content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        data = json.loads(text)
        return schema.model_validate(data)
