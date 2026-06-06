# OpsRAG SQL migrations

Plain SQL files applied by `opsrag.db.migrate`. The runner is intentionally
minimal so anyone reading these files knows exactly what hits the database.

## File format

```
NNNN_<snake_case_slug>.sql
```

- `NNNN` is a 4-digit zero-padded sequence number. Lex sort is the source
  of truth for apply order, so always pad -- `0010` not `10`.
- The slug is informational; the runner only cares about the prefix.

## Rules

1. **Never edit a migration after it has been applied to any environment.**
   The runner records sha256 of every file it applies and refuses to start
   if a recorded file's checksum no longer matches on disk. If you need
   to change schema, write a NEW migration.
2. **All migrations must be idempotent.** Use `CREATE TABLE IF NOT EXISTS`,
   `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`, etc. This makes re-running
   safe and lets us pre-create tables in tests without conflict.
3. **One logical change per file.** Keep migrations small so a failed
   apply leaves an obvious diff to fix.
4. **No DDL + DML in the same file** unless the DML is essential to the
   schema change (e.g. backfilling a NOT NULL column). Long backfills
   belong in a separate one-shot job.

## Running

```bash
# Apply everything pending:
python -m opsrag.db.migrate up

# Show what's applied vs pending:
python -m opsrag.db.migrate status

# Dry-run without touching the DB:
python -m opsrag.db.migrate up --dry-run

# Backup / restore wrappers:
python -m opsrag.db.migrate backup
python -m opsrag.db.migrate restore ./backups/opsrag-20260514-120000.sql.gz
```

DSN is read from `OPSRAG_POSTGRES_DSN` (preferred) or `POSTGRES_DSN`.
