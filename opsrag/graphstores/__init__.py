"""Knowledge graph store implementations.

The default backend is ``NullGraphStore`` (FR-019). ``Neo4jGraphStore``
ships as an optional alternative selected via
``knowledge_graph.provider: neo4j`` in ``config.yaml``.
"""
from opsrag.graphstores.neo4j import Neo4jGraphStore
from opsrag.graphstores.null import NullGraphStore

__all__ = ["Neo4jGraphStore", "NullGraphStore"]
