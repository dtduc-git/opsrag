"""Mem0-backed operational memory store (per-service).

Implements the `MemoryStore` Protocol (``opsrag/interfaces/memory.py``) so it
slots straight into ``providers.memory_store`` and rides the existing
``load_memory`` / ``save_memory`` graph nodes with no interface change.

Scope (per design "DESIGN 2 -- cache -> ServiceMemory only"): this is ONLY
per-service operational memory. The QA vector cache (``qa_cache*.py``) is kept
as-is and is NOT touched here.

Design constraints baked in:

* Backed by mem0 OSS ``Memory`` configured via ``Memory.from_config`` to REUSE
  the project's existing Qdrant (same URL, a dedicated ``mem0_collection``) and
  the project's configured LLM + embedder. The graph store stays OFF.
* The Protocol's ``namespace: tuple[str, ...]`` is mapped to a mem0 ``user_id``
  string via ``":".join(namespace)`` (e.g. ``("ops", "acme-notes-be")`` ->
  ``"ops:acme-notes-be"``).
* Null / empty-service safety: we refuse to write to a global / shared bucket.
  A namespace must contain a non-empty trailing "service" segment, otherwise
  the write is skipped (best-effort, logged).
* A small PII redaction pass (emails / bearer tokens) runs before ``.add()``.
* ALL reads / writes are best-effort: exceptions are swallowed + logged and
  NEVER propagate into the agent path.
* ``infer`` comes from config (``memory.mem0_infer``).
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from opsrag.interfaces.memory import Memory

if TYPE_CHECKING:  # pragma: no cover - typing only
    from opsrag.config import OpsRAGConfig

_log = logging.getLogger("opsrag.memory.mem0")

# --- PII redaction --------------------------------------------------------
# Conservative, dependency-free redaction applied to free text before it is
# handed to mem0's fact-extraction / storage. We are intentionally narrow:
# emails and obvious bearer / API tokens. Operational memory should never
# accumulate raw credentials or personal email addresses.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_BEARER_RE = re.compile(r"(?i)\b(bearer|token|api[_-]?key)\b[\s:=]+\S+")
# Long opaque secrets (e.g. ghp_..., sk-..., AKIA..., 32+ hex/base64-ish runs)
_SECRET_RE = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{12,}|[A-Fa-f0-9]{32,})\b"
)

_EMAIL_REDACTION = "[redacted-email]"
_TOKEN_REDACTION = "[redacted-token]"


def _redact_pii(text: str) -> str:
    """Strip emails and token-shaped secrets from a free-text string."""
    if not text:
        return text
    text = _EMAIL_RE.sub(_EMAIL_REDACTION, text)
    text = _BEARER_RE.sub(_TOKEN_REDACTION, text)
    text = _SECRET_RE.sub(_TOKEN_REDACTION, text)
    return text


def _redact_value(value: Any) -> Any:
    """Recursively redact PII in strings nested in dict/list/scalar values."""
    if isinstance(value, str):
        return _redact_pii(value)
    if isinstance(value, dict):
        return {k: _redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(v) for v in value]
    return value


# --- namespace / service handling ----------------------------------------

def _namespace_to_user_id(namespace: tuple[str, ...]) -> str:
    """Map a Protocol namespace tuple to a mem0 ``user_id`` string.

    ``("ops", "acme-notes-be")`` -> ``"ops:acme-notes-be"``.
    """
    return ":".join(str(seg) for seg in namespace)


def _has_service_segment(namespace: tuple[str, ...]) -> bool:
    """Null/empty-service guard.

    Refuse to address a global / shared bucket: require a namespace with at
    least one segment AND a non-empty trailing segment (the "service"). An
    empty tuple, or a tuple whose last segment is empty/whitespace, is treated
    as "no service" and writes are skipped.
    """
    if not namespace:
        return False
    last = namespace[-1]
    return bool(last) and bool(str(last).strip())


class Mem0ServiceMemory:
    """`MemoryStore` Protocol implementation backed by mem0 OSS ``Memory``.

    Best-effort: every method swallows + logs exceptions and never raises into
    the agent path. Writes to a service-less (global) namespace are skipped.
    """

    def __init__(self, memory: Any, *, infer: bool = True) -> None:
        # `memory` is a mem0 `Memory` instance (or any object exposing
        # add/search/get_all/delete). Injected so it can be mocked in tests.
        self._memory = memory
        self._infer = infer

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _result_to_memory(
        namespace: tuple[str, ...], item: dict
    ) -> Memory:
        """Adapt one mem0 result row into our `Memory` dataclass."""
        now = datetime.now(UTC)
        created = item.get("created_at") or now
        updated = item.get("updated_at") or created or now
        meta = item.get("metadata") or {}
        # Prefer an explicit logical key stored in metadata; fall back to the
        # mem0 row id so callers always get a stable, non-empty key.
        key = meta.get("_key") or item.get("id") or ""
        value = {
            "memory": item.get("memory"),
            "metadata": {k: v for k, v in meta.items() if k != "_key"},
        }
        if "score" in item:
            value["score"] = item["score"]
        return Memory(
            key=str(key),
            namespace=namespace,
            value=value,
            created_at=created if isinstance(created, datetime) else now,
            updated_at=updated if isinstance(updated, datetime) else now,
        )

    # -- MemoryStore Protocol ---------------------------------------------

    async def put(
        self, namespace: tuple[str, ...], key: str, value: dict
    ) -> None:
        if not _has_service_segment(namespace):
            _log.debug(
                "mem0 put skipped: no service segment in namespace %r", namespace
            )
            return
        user_id = _namespace_to_user_id(namespace)
        try:
            safe_value = _redact_value(value)
            # Build a compact message body from the value. We redact again on
            # the rendered text to catch PII that only appears once joined.
            text = _redact_pii(_render_value_text(safe_value))
            messages = [{"role": "user", "content": text}]
            metadata = {"_key": str(key)}
            if isinstance(safe_value, dict):
                # Carry scalar fields through as searchable metadata.
                for k, v in safe_value.items():
                    if isinstance(v, (str, int, float, bool)):
                        metadata[k] = v
            # mem0's client is SYNCHRONOUS and infer=True makes a blocking LLM
            # call -- run it off the event loop or it starves the request loop
            # (symptom: anyio WouldBlock / CancelledError -> 500 on /query).
            await asyncio.to_thread(
                lambda: self._memory.add(
                    messages,
                    user_id=user_id,
                    metadata=metadata,
                    infer=self._infer,
                )
            )
        except asyncio.CancelledError:
            raise
        except BaseException:  # noqa: BLE001 -- memory is best-effort; mem0 can
            # raise non-Exception failures (e.g. spaCy model-download SystemExit
            # in slim images). NEVER let memory break the agent.  # best-effort: never raise into the agent
            _log.warning("mem0 put failed for %s/%s", user_id, key, exc_info=True)

    async def get(
        self, namespace: tuple[str, ...], key: str
    ) -> Memory | None:
        if not _has_service_segment(namespace):
            return None
        user_id = _namespace_to_user_id(namespace)
        try:
            res = await asyncio.to_thread(
                lambda: self._memory.get_all(filters={"user_id": user_id})
            )
            rows = _extract_results(res)
            for item in rows:
                meta = item.get("metadata") or {}
                if str(meta.get("_key")) == str(key):
                    return self._result_to_memory(namespace, item)
            return None
        except asyncio.CancelledError:
            raise
        except BaseException:  # noqa: BLE001 -- memory is best-effort; mem0 can
            # raise non-Exception failures (e.g. spaCy model-download SystemExit
            # in slim images). NEVER let memory break the agent.
            _log.warning("mem0 get failed for %s/%s", user_id, key, exc_info=True)
            return None

    async def search(
        self,
        namespace: tuple[str, ...],
        query: str | None = None,
        limit: int = 10,
    ) -> list[Memory]:
        if not _has_service_segment(namespace):
            return []
        user_id = _namespace_to_user_id(namespace)
        try:
            if query:
                res = await asyncio.to_thread(
                    lambda: self._memory.search(
                        query, filters={"user_id": user_id}, top_k=limit
                    )
                )
            else:
                res = await asyncio.to_thread(
                    lambda: self._memory.get_all(
                        filters={"user_id": user_id}, top_k=limit
                    )
                )
            rows = _extract_results(res)
            return [self._result_to_memory(namespace, item) for item in rows[:limit]]
        except asyncio.CancelledError:
            raise
        except BaseException:  # noqa: BLE001 -- memory is best-effort; mem0 can
            # raise non-Exception failures (e.g. spaCy model-download SystemExit
            # in slim images). NEVER let memory break the agent.
            _log.warning("mem0 search failed for %s", user_id, exc_info=True)
            return []

    async def delete(self, namespace: tuple[str, ...], key: str) -> bool:
        if not _has_service_segment(namespace):
            return False
        user_id = _namespace_to_user_id(namespace)
        try:
            res = await asyncio.to_thread(
                lambda: self._memory.get_all(filters={"user_id": user_id})
            )
            rows = _extract_results(res)
            deleted = False
            for item in rows:
                meta = item.get("metadata") or {}
                if str(meta.get("_key")) == str(key) and item.get("id"):
                    await asyncio.to_thread(
                        lambda mid=item["id"]: self._memory.delete(memory_id=mid)
                    )
                    deleted = True
            return deleted
        except asyncio.CancelledError:
            raise
        except BaseException:  # noqa: BLE001 -- memory is best-effort; mem0 can
            # raise non-Exception failures (e.g. spaCy model-download SystemExit
            # in slim images). NEVER let memory break the agent.
            _log.warning("mem0 delete failed for %s/%s", user_id, key, exc_info=True)
            return False


def _render_value_text(value: Any) -> str:
    """Render a put() value dict into a short natural-language line for mem0.

    mem0 stores/extracts from message text; a readable line gives infer mode
    better material than a raw repr while staying deterministic for infer=False.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = [f"{k}: {v}" for k, v in value.items()]
        return "; ".join(parts)
    return str(value)


def _extract_results(res: Any) -> list[dict]:
    """mem0 v2 returns ``{"results": [...]}``; tolerate a bare list too."""
    if isinstance(res, dict):
        rows = res.get("results", [])
    else:
        rows = res
    return list(rows) if rows else []


# --- factory helper -------------------------------------------------------

# Map opsrag provider names -> mem0 provider names for LLM + embedder.
_LLM_PROVIDER_MAP = {
    "anthropic": "anthropic",
    "openai": "openai",
    "vertex": "gemini",
    "bedrock": "aws_bedrock",
    "ollama": "ollama",
}
_EMBED_PROVIDER_MAP = {
    "openai": "openai",
    "vertex": "vertexai",
    "bedrock": "aws_bedrock",
    "fastembed": "fastembed",
    "ollama": "ollama",
    "cohere": "openai",  # mem0 has no cohere embedder; caller should override
}


def build_mem0_store(
    settings: OpsRAGConfig,
    qdrant_client_or_url: Any,
    llm_cfg: Any,
    embed_cfg: Any,
) -> Mem0ServiceMemory:
    """Build a `Mem0ServiceMemory` reusing the project's Qdrant + LLM + embedder.

    Parameters
    ----------
    settings:
        The root `OpsRAGConfig`; ``settings.memory`` supplies
        ``mem0_collection`` and ``mem0_infer``, and ``settings.vector_store``
        supplies the Qdrant URL / api-key fallback.
    qdrant_client_or_url:
        Either a live ``qdrant_client.QdrantClient`` (preferred -- reuses the
        existing connection) or a URL string. Passed straight into mem0's
        qdrant ``config`` as ``client`` or ``url`` respectively.
    llm_cfg:
        The project's `LLMConfig` (``settings.llm``).
    embed_cfg:
        The project's `EmbeddingConfig` (``settings.embedding``).

    The graph store is left OFF (mem0 defaults to no graph store when the
    ``graph_store`` key is omitted).
    """
    from mem0 import Memory

    mem_cfg = settings.memory
    vs_cfg = settings.vector_store

    # --- mem0 embedder: optional override of the main retrieval embedder ---
    # The main embedder may be a code-tuned model mem0 can't drive (e.g. Cohere
    # Embed v4 on Bedrock -> mem0's aws_bedrock embedder sends Titan-style
    # payloads -> "Malformed request"). memory.mem0_embed_* lets the operator
    # point mem0 at a simpler, compatible embedder (memory facts are short NL).
    mem_embed_provider_name = (
        mem_cfg.mem0_embed_provider or getattr(embed_cfg, "provider", "openai")
    )
    mem_embed_model = mem_cfg.mem0_embed_model or getattr(embed_cfg, "model", None)
    mem_embed_dim = mem_cfg.mem0_embed_dimension or getattr(embed_cfg, "dimension", None)

    # --- vector store: reuse existing Qdrant -----------------------------
    qdrant_config: dict[str, Any] = {
        "collection_name": mem_cfg.mem0_collection,
    }
    if mem_embed_dim:
        qdrant_config["embedding_model_dims"] = mem_embed_dim
    # Prefer an injected live client; else fall back to a URL.
    if isinstance(qdrant_client_or_url, str):
        qdrant_config["url"] = qdrant_client_or_url
    elif qdrant_client_or_url is not None:
        qdrant_config["client"] = qdrant_client_or_url
    else:
        qdrant_config["url"] = getattr(vs_cfg, "url", "http://localhost:6333")

    # --- llm: reuse the project's configured backend ---------------------
    llm_provider = _LLM_PROVIDER_MAP.get(
        getattr(llm_cfg, "provider", "anthropic"), "anthropic"
    )
    llm_inner: dict[str, Any] = {}
    if getattr(llm_cfg, "model", None):
        llm_inner["model"] = llm_cfg.model

    # --- embedder: mem0 override (memory.mem0_embed_*) or main embedder ---
    embed_provider = _EMBED_PROVIDER_MAP.get(mem_embed_provider_name, "openai")
    embed_inner: dict[str, Any] = {}
    if mem_embed_model:
        embed_inner["model"] = mem_embed_model
    if mem_embed_dim:
        embed_inner["embedding_dims"] = mem_embed_dim

    config_dict: dict[str, Any] = {
        "vector_store": {"provider": "qdrant", "config": qdrant_config},
        "llm": {"provider": llm_provider, "config": llm_inner},
        "embedder": {"provider": embed_provider, "config": embed_inner},
        # graph_store intentionally omitted -> OFF.
    }

    memory = Memory.from_config(config_dict)
    return Mem0ServiceMemory(memory, infer=mem_cfg.mem0_infer)


__all__ = ["Mem0ServiceMemory", "build_mem0_store"]
