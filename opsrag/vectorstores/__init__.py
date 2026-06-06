"""Vector store implementations."""
from opsrag.vectorstores.qdrant import QdrantVectorStore

__all__ = ["QdrantVectorStore"]

try:
    from opsrag.vectorstores.pgvector import PgVectorStore
    __all__.append("PgVectorStore")
except ImportError:
    pass  # asyncpg not installed -- pgvector store unavailable
