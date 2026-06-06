"""Protocol interfaces for all pluggable OpsRAG components.

Every stage of the pipeline is defined as a typing.Protocol so implementations
can be swapped via structural subtyping without inheritance.
"""
from opsrag.interfaces.chunker import Chunk, ChunkingStrategy
from opsrag.interfaces.embedder import EmbeddingProvider
from opsrag.interfaces.entity_extractor import EntityExtractor, ExtractionResult
from opsrag.interfaces.graphstore import (
    Entity,
    GraphSearchResult,
    KnowledgeGraphStore,
    Relationship,
)
from opsrag.interfaces.llm import LLMProvider, LLMResponse
from opsrag.interfaces.memory import Memory, MemoryStore
from opsrag.interfaces.observability import ObservabilityProvider
from opsrag.interfaces.parser import DocSection, DocType, DocumentParser, ParsedDocument
from opsrag.interfaces.reranker import Reranker, RerankResult
from opsrag.interfaces.scm import RepoFile, SCMProvider, WebhookEvent
from opsrag.interfaces.session import SessionStore
from opsrag.interfaces.vectorstore import SearchResult, VectorStore

__all__ = [
    "RepoFile", "WebhookEvent", "SCMProvider",
    "DocType", "DocSection", "ParsedDocument", "DocumentParser",
    "Chunk", "ChunkingStrategy",
    "EmbeddingProvider",
    "SearchResult", "VectorStore",
    "Entity", "Relationship", "GraphSearchResult", "KnowledgeGraphStore",
    "ExtractionResult", "EntityExtractor",
    "LLMResponse", "LLMProvider",
    "RerankResult", "Reranker",
    "SessionStore",
    "Memory", "MemoryStore",
    "ObservabilityProvider",
]
