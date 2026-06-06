# Acme Notes - Incident response runbook

**Service:** acme-notes (all components)
**Owner:** Acme Notes platform team
**Runbook type:** on-call / incident

## Overview

The first-responder checklist for any Acme Notes production incident. Use this
to triage before diving into a component-specific runbook.

## Prerequisites

- You are the current on-call, or you have been paged in.
- Access to the dashboards, the incident channel, and `kubectl`.

## Procedure - triage

1. Acknowledge the page and post "investigating" in the incident channel.
2. Check `/healthz` and `/readyz` for `acme-notes-api`.
3. Look at the four golden signals on the dashboard: latency, traffic, errors,
   saturation.
4. Identify the failing component:
   - API errors / latency -> [Scaling](003-acme-notes-scaling.md)
   - Database errors -> [DB failover](002-acme-notes-db-failover.md)
   - Bad recent release -> [Deploy & Rollback](001-acme-notes-deploy.md)

## Verification

- After mitigation, confirm error rate and latency return to baseline and stay
  there for 15 minutes before closing the incident.

## Rollback

If a recent deploy is the suspected cause, roll it back first and ask questions
later - `kubectl -n acme-notes rollout undo deploy/acme-notes-api`.

## Escalation

- Sev-1 (full outage): page the platform lead and the engineering manager.
- Sev-2 (degraded): keep the incident channel updated every 30 minutes.

## After the incident

Write a postmortem within two business days. See the examples under
`samples/postmortems/`.
