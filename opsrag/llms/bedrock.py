"""AWS Bedrock LLM provider -- Claude on Bedrock via the Converse API.

Uses the Bedrock Converse API which provides a unified interface for
Claude, Llama, Mistral, and other models hosted on Bedrock.

Auth via standard AWS credential chain:
  - Environment: AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY + AWS_DEFAULT_REGION
  - Shared credentials file (~/.aws/credentials)
  - IAM role (EC2/ECS/EKS)
  - SSO: aws sso login --profile your-profile

Requires: pip install boto3
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from pydantic import BaseModel

from opsrag.interfaces.llm import LLMResponse
from opsrag.llms.content import to_bedrock_content
from opsrag.llms.json_extract import extract_first_json_object

# boto3 is an OPTIONAL dependency (the `bedrock` extra). Importing it lazily in
# __init__ keeps this module -- and everything that imports it transitively, e.g.
# opsrag.api.server -- importable on a build WITHOUT boto3 (the default dev/CI
# install). It's only needed when a Bedrock provider is actually constructed.


class BedrockLLM:
    # Class-level cache of model ids that reject `temperature` in Converse
    # (e.g. Claude Opus 4.8 / Haiku 4.5). Populated lazily on the first
    # temperature ValidationException so the retry only ever happens once.
    _no_temperature: set[str] = set()

    def __init__(
        self,
        model: str = "anthropic.claude-sonnet-4-20250514-v1:0",
        region: str | None = None,
        profile: str | None = None,
        default_max_tokens: int = 4096,
        *,
        request_timeout: float | None = None,
        connect_timeout: float | None = None,
        max_retries: int | None = None,
    ):
        import boto3
        session = boto3.Session(
            region_name=region,
            profile_name=profile,
        )
        # Optional botocore-level robustness knobs. Left unset, boto3 keeps its
        # own native defaults so existing callers see identical behavior. When
        # any is provided, build a botocore Config -- mirrors the precedent in
        # opsrag/embedders/bedrock.py (adaptive retries + explicit timeouts).
        client_kwargs: dict[str, Any] = {}
        if (
            request_timeout is not None
            or connect_timeout is not None
            or max_retries is not None
        ):
            from botocore.config import Config as _BotoConfig
            cfg_kwargs: dict[str, Any] = {}
            if request_timeout is not None:
                cfg_kwargs["read_timeout"] = request_timeout
            if connect_timeout is not None:
                cfg_kwargs["connect_timeout"] = connect_timeout
            if max_retries is not None:
                cfg_kwargs["retries"] = {
                    "max_attempts": max_retries,
                    "mode": "adaptive",
                }
            client_kwargs["config"] = _BotoConfig(**cfg_kwargs)
        self._client = session.client("bedrock-runtime", **client_kwargs)
        self._model = model
        self._default_max_tokens = default_max_tokens

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
        # response_schema is a no-op for Bedrock Converse -- no native JSON
        # schema mode; structured outputs go through generate_structured().
        _ = response_schema
        start = time.perf_counter()

        converse_messages = []
        for msg in messages:
            converse_messages.append({
                "role": msg["role"],
                "content": to_bedrock_content(msg.get("content", "")),
            })

        inf_cfg: dict[str, Any] = {
            "maxTokens": max_tokens or self._default_max_tokens,
        }
        # Newer Claude models on Bedrock (Opus 4.8, Haiku 4.5, ...) reject
        # `temperature` ("deprecated for this model"). Omit it for models
        # we've learned don't support it; otherwise include it and, on a
        # temperature ValidationException, cache + retry without it.
        if self._model not in BedrockLLM._no_temperature:
            inf_cfg["temperature"] = temperature
        kwargs: dict[str, Any] = {
            "modelId": self._model,
            "messages": converse_messages,
            "inferenceConfig": inf_cfg,
        }

        if system_prompt:
            kwargs["system"] = [{"text": system_prompt}]

        # boto3 is synchronous; run it in a thread so a Bedrock call never
        # blocks the asyncio event loop (otherwise a single in-flight
        # generation freezes EVERY other request -- /me, /ui-config, etc.).
        try:
            resp = await asyncio.to_thread(self._client.converse, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if "temperature" in inf_cfg and "temperature" in str(exc).lower():
                BedrockLLM._no_temperature.add(self._model)
                inf_cfg.pop("temperature", None)
                resp = await asyncio.to_thread(self._client.converse, **kwargs)
            else:
                raise
        latency_ms = (time.perf_counter() - start) * 1000

        output = resp.get("output", {})
        content_blocks = output.get("message", {}).get("content", [])
        content = "".join(
            block.get("text", "") for block in content_blocks
        )

        usage = resp.get("usage", {})
        in_tok = usage.get("inputTokens", 0)
        out_tok = usage.get("outputTokens", 0)

        from opsrag.usage import tracker
        tracker.record(
            model=self._model,
            input_tokens=in_tok, output_tokens=out_tok, latency_ms=latency_ms,
            purpose=purpose,
        )

        return LLMResponse(
            content=content,
            model=self._model,
            usage={"input_tokens": in_tok, "output_tokens": out_tok},
            latency_ms=latency_ms,
            raw_response=None,  # Don't hold boto3 response in memory
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
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise TypeError("schema must be a pydantic.BaseModel subclass")

        json_schema = schema.model_json_schema()
        instruction = (
            "Respond ONLY with a single JSON object that matches this schema. "
            "No prose, no code fences.\n\n"
            f"Schema:\n{json.dumps(json_schema, indent=2)}"
        )
        system = f"{system_prompt}\n\n{instruction}" if system_prompt else instruction

        # max_tokens caps output; left None it preserves generate's default-4096
        # behavior so existing callers are unchanged.
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
