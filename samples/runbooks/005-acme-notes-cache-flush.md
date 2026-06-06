# Acme Notes - Cache flush runbook

**Service:** acme-notes-cache (Redis)
**Owner:** Acme Notes platform team
**Runbook type:** cache / maintenance

## Overview

How to safely flush the Acme Notes Redis cache when stale data is being served,
without taking the product down. The cache stores rendered note previews and
session lookups.

## Prerequisites

- Confirm the symptom is stale cached data (e.g. an edited note still shows the
  old body), not a database problem.
- Know the Redis endpoint and auth from the secret store.

## Procedure - selective flush (preferred)

1. Identify the affected key prefix (e.g. `note:preview:<id>`).
2. Delete only those keys:
   `redis-cli --scan --pattern 'note:preview:*' | xargs redis-cli del`.
3. The API repopulates the cache on the next read.

## Procedure - full flush (last resort)

1. Announce a brief latency bump in the incident channel.
2. `redis-cli flushdb` against the Acme Notes cache database only.
3. Expect elevated latency for ~1 minute while the cache warms.

## Verification

- The previously-stale note now renders its current body.
- Cache hit rate recovers on the dashboard within a few minutes.

## Rollback

There is no rollback for a flush - the cache simply rewarms from the database.
If latency stays high after a full flush, the database may be the real
bottleneck; see [DB failover](002-acme-notes-db-failover.md).

## Troubleshooting

If keys reappear stale immediately after flushing, a writer is populating bad
data - stop the offending job before flushing again.
