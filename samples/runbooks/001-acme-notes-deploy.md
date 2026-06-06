# Acme Notes - Deploy & Rollback Runbook

**Service:** acme-notes-api
**Owner:** Acme Notes platform team
**Runbook type:** deployment

## Overview

This runbook covers a standard production deployment of the Acme Notes API
and how to roll it back if the release misbehaves. Acme Notes is a fictional
note-taking SaaS used only as sample data for opsrag.

## Prerequisites

- You are on-call or have change-approval for the `acme-notes` namespace.
- The release image is published to the registry and tagged (e.g. `v1.8.2`).
- `kubectl` context points at the production cluster `acme-prod`.

## Procedure - deploy

1. Confirm the target image tag exists.
2. Update the Deployment image:
   `kubectl -n acme-notes set image deploy/acme-notes-api api=registry.example.com/acme-notes-api:v1.8.2`
3. Watch the rollout: `kubectl -n acme-notes rollout status deploy/acme-notes-api`.
4. The Deployment uses a RollingUpdate strategy (maxSurge 1, maxUnavailable 0),
   so traffic is never dropped during a healthy rollout.

## Verification

- `kubectl -n acme-notes get pods -l app=acme-notes-api` shows all pods Ready.
- `curl -sf https://api.acme-notes.example.com/healthz` returns 200.
- Error rate in the dashboard stays under 0.5% for five minutes.

## Rollback

If the new release raises errors or latency:

1. Roll back to the previous ReplicaSet:
   `kubectl -n acme-notes rollout undo deploy/acme-notes-api`
2. Watch `kubectl -n acme-notes rollout status deploy/acme-notes-api`.
3. Verify health as above.
4. If a database migration shipped with the bad release, see
   [DB failover & rollback](002-acme-notes-db-failover.md) before declaring
   the rollback complete.

## Escalation

If rollback does not restore health within 15 minutes, page the Acme Notes
platform on-call and open an incident.
