# Postmortem - Acme Notes API latency spike (2026-02-03)

**Severity:** Sev-2
**Duration:** 1 hour 10 minutes
**Author:** Acme Notes platform team

## Summary

The Acme Notes API p95 latency rose from ~180 ms to over 2 s for just over an
hour after a deploy that introduced an N+1 query on the notes-list endpoint.

## Impact

- The notes list felt sluggish for most active users.
- No data loss; no failed writes.
- Latency SLO (300 ms p95) was breached for 70 minutes.

## Timeline

- 13:02 - Release `v1.7.0` deployed.
- 13:09 - p95 latency alert fires.
- 13:20 - On-call correlates the spike with the 13:02 deploy.
- 13:34 - Rollback to `v1.6.4` started.
- 13:41 - Latency recovers to baseline.
- 14:12 - Incident closed after a clean monitoring window.

## Detection

The p95 latency alert worked as intended and fired within 7 minutes of the
regression.

## Root cause

A refactor of the notes-list endpoint removed an eager-load, turning one query
into one-per-note. Under real note counts this produced hundreds of queries per
request.

## Mitigation

Rolled back to the previous release following the
[Deploy & Rollback](../runbooks/001-acme-notes-deploy.md) runbook. The fix
(restoring the eager-load) shipped two days later with a regression test.

## Action items

- Add a query-count assertion to the notes-list endpoint test.
- Add a pre-deploy load test for the top three endpoints.

## Lessons learned

Rollback-first was the right call and kept impact to ~1 hour. The regression
should have been caught by a load test before release.
