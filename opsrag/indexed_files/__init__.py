"""Idempotent indexing -- track (repo, branch, path) -> content_hash so
subsequent indexing runs skip unchanged files."""
