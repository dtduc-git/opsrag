# Acme Notes - Scaling the API tier

**Service:** acme-notes-api
**Owner:** Acme Notes platform team
**Runbook type:** capacity

## Overview

This runbook describes how to scale the Acme Notes API up or down, both
manually and via the HorizontalPodAutoscaler, when traffic changes.

## Prerequisites

- Confirm the bottleneck is the API tier (CPU/latency), not the database.
- Check current replica count:
  `kubectl -n acme-notes get deploy acme-notes-api`.

## Procedure - manual scale

1. Scale to the desired replica count:
   `kubectl -n acme-notes scale deploy/acme-notes-api --replicas=8`.
2. Watch pods become Ready.
3. Confirm latency recovers on the dashboard.

## Procedure - autoscaling

The API has an HPA targeting 70% CPU, min 3 / max 12 replicas. To adjust:

1. Edit the HPA: `kubectl -n acme-notes edit hpa acme-notes-api`.
2. Change `minReplicas` / `maxReplicas` / target utilization as needed.
3. The autoscaler reacts within ~1 minute.

## Verification

- p95 request latency returns under the 300 ms SLO.
- No pods are in `Pending` (if they are, the cluster is out of capacity -
  scale the node pool first).

## Rollback

To return to baseline, set replicas back to the HPA `minReplicas` or let the
autoscaler scale down naturally after traffic subsides.

## Troubleshooting

If scaling up does not reduce latency, the bottleneck is likely downstream
(database or cache) - see [DB failover](002-acme-notes-db-failover.md) and the
cache runbook.
