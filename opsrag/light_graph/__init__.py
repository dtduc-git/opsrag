"""Lightweight entity-graph (Postgres adjacency) for the entity-expansion
retrieval lane. NOT Neo4j, NOT the main retrieval line -- a tiny edges table
used for a 1-hop expand AFTER vector search. See opsrag/db/migrations/0008."""
from opsrag.light_graph.postgres import LightGraphStore

__all__ = ["LightGraphStore"]
