"""OpsRAG database tooling -- migration runner + SQL migrations.

This package is intentionally thin: the runner in ``migrate`` discovers
``migrations/000N_<name>.sql`` files and applies them in lex order,
recording each in ``opsrag_schema_migrations``. New tables / ALTERs
go here going forward; legacy ``CREATE TABLE IF NOT EXISTS`` blocks
inside their owning modules (e.g. ``usage_persistence``) stay put.
"""
