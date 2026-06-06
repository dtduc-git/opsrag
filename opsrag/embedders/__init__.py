"""Embedding provider implementations."""
from opsrag.embedders.openai import OpenAIEmbeddings

__all__ = ["OpenAIEmbeddings"]

try:
    from opsrag.embedders.fastembed import FastEmbedEmbeddings
    __all__.append("FastEmbedEmbeddings")
except ImportError:
    pass

try:
    from opsrag.embedders.vertex import VertexAIEmbeddings
    __all__.append("VertexAIEmbeddings")
except ImportError:
    pass

try:
    from opsrag.embedders.bedrock import BedrockEmbeddings
    __all__.append("BedrockEmbeddings")
except ImportError:
    pass
