"""Vertex Gemini judge adapter for DeepEval -- Path B (zero LangChain).

Bridges DeepEval's `DeepEvalBaseLLM` interface to the Vertex AI Gemini SDK
directly, so eval calls share the same auth (ADC), quota project, and
region as production. No `langchain_google_genai` involved.

Implements the four required methods (load_model / generate / a_generate /
get_model_name) plus generate_schema / a_generate_schema for metrics that
expect structured output (FaithfulnessMetric, GEval, HallucinationMetric).
"""
from __future__ import annotations

import json
import logging
from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import BaseModel
from vertexai.generative_models import GenerationConfig, GenerativeModel

from opsrag.llms.vertex import _extract_gemini_text

_log = logging.getLogger("opsrag.eval.vertex_judge")


class VertexGeminiJudge(DeepEvalBaseLLM):
    """LLM-as-judge adapter using Vertex Gemini directly.

    Default model: gemini-2.5-pro (slower but more accurate for judge tasks).
    Override via constructor for cheaper iteration during development.
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        self.model_name = model_name
        self._model: GenerativeModel | None = None

    def load_model(self) -> GenerativeModel:
        if self._model is None:
            self._model = GenerativeModel(self.model_name)
        return self._model

    def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> Any:
        """Sync generate. If `schema` provided, returns parsed Pydantic instance."""
        if schema is not None:
            return self._sync_generate_schema(prompt, schema)
        resp = self.load_model().generate_content(prompt)
        return _extract_gemini_text(resp)

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None) -> Any:
        """Async generate. If `schema` provided, returns parsed Pydantic instance."""
        if schema is not None:
            return await self._async_generate_schema(prompt, schema)
        resp = await self.load_model().generate_content_async(prompt)
        return _extract_gemini_text(resp)

    # Some DeepEval metric versions look for explicit *_schema methods.
    def generate_schema(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        return self._sync_generate_schema(prompt, schema)

    async def a_generate_schema(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        return await self._async_generate_schema(prompt, schema)

    def _sync_generate_schema(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        cfg = GenerationConfig(
            response_mime_type="application/json",
            response_schema=schema.model_json_schema(),
            temperature=0.0,
        )
        resp = self.load_model().generate_content(prompt, generation_config=cfg)
        return _parse_json_to_schema(_extract_gemini_text(resp), schema)

    async def _async_generate_schema(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        cfg = GenerationConfig(
            response_mime_type="application/json",
            response_schema=schema.model_json_schema(),
            temperature=0.0,
        )
        resp = await self.load_model().generate_content_async(prompt, generation_config=cfg)
        return _parse_json_to_schema(_extract_gemini_text(resp), schema)

    def get_model_name(self) -> str:
        return f"vertex-{self.model_name}"


def _parse_json_to_schema(text: str, schema: type[BaseModel]) -> BaseModel:
    """Parse Gemini JSON output to a Pydantic schema, with a markdown-fence fallback."""
    cleaned = text.strip()
    # Defensive: even with response_mime_type=application/json, some model
    # versions occasionally wrap output in ```json ... ``` fences.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        _log.warning("vertex judge returned non-JSON; raw=%s", text[:200])
        raise
    return schema.model_validate(data)
