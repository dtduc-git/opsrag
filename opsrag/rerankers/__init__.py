"""Reranker implementations."""
from opsrag.rerankers.cohere import CohereReranker
from opsrag.rerankers.noop import NoOpReranker

__all__ = ["CohereReranker", "NoOpReranker"]

try:
    from opsrag.rerankers.fastembed_reranker import FastEmbedReranker
    __all__.append("FastEmbedReranker")
except ImportError:
    pass

try:
    from opsrag.rerankers.vertex import VertexReranker
    __all__.append("VertexReranker")
except ImportError:
    pass
