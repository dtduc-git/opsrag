# Acme Notes - DB failover & rollback

**Service:** acme-notes-db (PostgreSQL, primary + replica)
**Owner:** Acme Notes platform team
**Runbook type:** database / disaster recovery

## Overview

This runbook explains how to fail the Acme Notes database over to its standby
replica and how to roll back a bad schema migration. Acme Notes runs a primary
PostgreSQL instance with one synchronous standby replica.

## Prerequisites

- Confirm the primary is actually unhealthy (not a transient network blip)
  using the database dashboard and `pg_isready`.
- You have the `acme-notes-db` admin credentials from the secret store.
- Announce the failover in the incident channel before starting.

## Procedure - failover to replica

1. Fence the failing primary so it cannot accept writes.
2. Promote the standby: `pg_ctl promote -D /var/lib/postgresql/data`.
3. Update the `acme-notes-db` connection secret to point the application at
   the newly promoted instance.
4. Restart the API so it picks up the new connection:
   `kubectl -n acme-notes rollout restart deploy/acme-notes-api`.

## Procedure - roll back a bad migration

1. Stop the API to prevent further writes against the bad schema.
2. Restore the pre-migration snapshot (snapshots are taken automatically
   before every migration job runs).
3. Re-point the application at the restored instance.
4. Re-deploy the previous application image - see
   [Deploy & Rollback](001-acme-notes-deploy.md).

## Verification

- `pg_isready -h <new-primary>` returns `accepting connections`.
- The API `/readyz` endpoint returns 200.
- Read and write a canary note through the API and confirm it persists.

## Rollback (of the failover itself)

Once the original primary is repaired, re-attach it as a standby and let it
catch up before considering any switchback. Never switch back during an active
incident.

## Escalation

If promotion fails or replication lag exceeds the configured threshold, page
the database on-call immediately.
