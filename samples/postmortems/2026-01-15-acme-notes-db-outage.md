# Postmortem - Acme Notes database outage (2026-01-15)

**Severity:** Sev-1
**Duration:** 42 minutes
**Author:** Acme Notes platform team

## Summary

The Acme Notes primary database ran out of disk and stopped accepting writes,
making the product read-only-then-unavailable for 42 minutes during the
morning peak.

## Impact

- 100% of write requests failed for 28 minutes (note creation and edits).
- Read traffic degraded as the cache expired.
- Roughly 18,000 users encountered errors.

## Timeline

- 08:14 - Disk usage on the primary crosses 95%; no alert fires (threshold was
  set at 98%).
- 08:31 - Writes begin failing; error rate alert pages on-call.
- 08:36 - On-call identifies the full disk via the database dashboard.
- 08:44 - Old WAL segments archived and removed to free space.
- 08:58 - Writes recover; cache rewarms.
- 09:13 - Error rate back to baseline; incident closed.

## Detection

The disk-usage alert threshold (98%) was too high to give useful lead time. The
incident was effectively detected by the downstream write-error alert.

## Root cause

A backfill job wrote far more WAL than usual, and WAL archiving had silently
fallen behind, so segments accumulated on the primary's data disk until it
filled.

## Mitigation

Archived and pruned old WAL segments, then confirmed archiving resumed. No
failover was needed because the disk was recoverable in place. See
[DB failover & rollback](../runbooks/002-acme-notes-db-failover.md).

## Action items

- Lower the disk-usage alert to 85% with a separate page at 92%.
- Add monitoring on WAL archive lag specifically.
- Rate-limit backfill jobs so they cannot outrun WAL archiving.

## Lessons learned

What went well: once detected, recovery was fast. What went wrong: the alert
thresholds gave no early warning, and WAL archive lag was invisible.
