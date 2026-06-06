"""Resolve a free-form alert / Rootly URL into a structured payload.

Today's investigation entry points let the user paste either:
  - A Rootly alert URL (`https://rootly.com/account/alerts/<short_id>`),
    in which case we fetch by short_id directly.
  - A pasted alert title from Slack like
    `[P2][prod/k8s/none] [Internal][Istio] High 500 requests are more than 14.29% from acme-notes-be-worker to service acme-appservice-main`,
    in which case we list the recent Rootly alerts and fuzzy-match on title.

When neither path resolves to a Rootly alert (or Rootly is down) the
caller falls back to the existing hybrid regex+LLM alert_extractor.

The structured Rootly payload gives us:
  - env, namespace, service, source/destination workload
  - the actual PromQL expression (annotations.expr)
  - the runbook URL (annotations.runbook_url)
  - the firing value, severity, status

Reference: payload shape verified 2026-05-13 against alerts/6aEl4t.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

_log = logging.getLogger("opsrag.agents.investigation.rootly_resolver")


ROOTLY_API_BASE = "https://api.rootly.com/v1"
_TOKEN_ENV_KEYS = ("OPSRAG_ROOTLY_TOKEN", "ROOTLY_API_TOKEN")
# Match a Rootly alert URL on either a tenant subdomain or rootly.com.
_ROOTLY_URL_RE = re.compile(
    r"https?://(?:[a-z0-9-]+\.)?rootly\.com/(?:account/)?alerts/([A-Za-z0-9_-]{4,32})\b"
)
# Title fuzzy-match: recency window in seconds (12h -- see resolver design doc).
_DEFAULT_RECENCY_SEC = 12 * 3600
# Jaccard threshold below which we consider "no match".
_DEFAULT_JACCARD_THRESHOLD = 0.55
# Tokens to ignore when scoring title similarity. Common SRE alert
# boilerplate that's not discriminative.
_TITLE_STOPWORDS: set[str] = {
    "the", "is", "to", "from", "are", "of", "a", "an", "than", "more",
    "service", "internal", "external", "public", "private", "high", "low",
    "requests", "request", "error", "rate", "errors",
    # bracketed channel/severity prefixes survive normalisation but should
    # not dominate scoring
    "p1", "p2", "p3", "p4", "prod", "staging", "shared", "dev", "k8s", "none",
}


@dataclass
class AlertEnrichment:
    """Structured fields the investigation graph needs, plus provenance."""
    summary: str = ""
    service: str | None = None
    namespace: str | None = None
    env: str | None = None
    source_workload: str | None = None
    destination_workload: str | None = None
    promql_expression: str | None = None
    runbook_url: str | None = None
    severity: str | None = None
    status: str | None = None
    firing_value: str | None = None
    generator_url: str | None = None
    short_id: str | None = None
    match_source: str = ""  # "rootly:url" | "rootly:match:<short_id>/0.78" | ""
    match_score: float | None = None

    def is_useful(self) -> bool:
        """Did we recover anything beyond the raw text?"""
        return bool(self.service or self.env or self.namespace or self.promql_expression)


# Module-level state for the cached recent-alerts fetch. One request per
# investigation typically triggers the list; cache prevents storms when
# the user retries inside a minute.
_recent_cache: dict[str, Any] = {"ts": 0.0, "items": []}
_RECENT_CACHE_TTL_SEC = 60


def _resolve_token() -> str | None:
    for key in _TOKEN_ENV_KEYS:
        val = os.environ.get(key)
        if val and val.strip():
            return val.strip()
    return None


def _normalize_title(s: str) -> set[str]:
    """Lowercase + strip `[Pn]`/`[Tag]` brackets + tokenise + drop stopwords."""
    if not s:
        return set()
    low = s.lower()
    # Drop bracketed prefix tags ([P2], [Public], [prod/k8s/none], etc.)
    low = re.sub(r"\[[^\]]*\]", " ", low)
    # Replace non-word chars with spaces (but keep `-` since it's part of
    # service names like `acme-notes-be-worker`).
    low = re.sub(r"[^a-z0-9\- ]+", " ", low)
    toks = {t for t in low.split() if t and t not in _TITLE_STOPWORDS and len(t) > 1}
    return toks


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def parse_rootly_short_id(text: str) -> str | None:
    """Pull a Rootly alert short_id from a URL embedded in `text`."""
    if not text:
        return None
    m = _ROOTLY_URL_RE.search(text)
    return m.group(1) if m else None


def _ns_from_summary(summary: str) -> str | None:
    """Some alert templates use `[<env>/<tier>/<ns>]` in the summary
    even when the labels lack `namespace`. Extract as a backup."""
    m = re.search(r"\[\s*[a-z]{2,12}\s*/\s*[a-z0-9_-]+\s*/\s*([a-z0-9_-]+)\s*\]", summary or "", re.IGNORECASE)
    if not m:
        return None
    ns = m.group(1).lower()
    return None if ns in ("none", "n/a", "-", "unknown") else ns


# Namespaces that hold monitoring sidecars / cluster infra, NOT the
# failing app. When `labels.namespace` matches one of these, we look
# elsewhere for the real service.
_INFRA_NAMESPACES = {"monitoring", "none", "kube-system", "kube-public", "default"}


def _derive_service_and_namespace(labels: dict, summary: str) -> tuple[str | None, str | None]:
    """Class-aware mapping from Rootly/AlertManager labels -> (service, namespace).

    Survey of 20 recent prod alerts (2026-05-13) showed `alertrule_appname`
    is ALWAYS the helm chart that defined the rule (`alerting-cloudsql`,
    `alerting-http-probes`, `alerting-k8s`, ...) -- never the failing service.
    The failing service lives in different labels depending on the alert
    class.

    Priority order (highest -> lowest):
      1. Istio inter-service 5xx -> `destination_workload`
      2. HTTP probe / saas-outage -> `app_instance`
      3. CloudSQL alerts -> `datname` (common convention: DB == service)
      4. Per-container alerts (e.g. ES index size) -> `container`
      5. Namespace-scoped app alerts -> `namespace` (when not infra)
      6. Summary tag `[<env>/<tier>/<ns>]` when ns is meaningful
      7. Last resort -> `alertrule_release` / `alertrule_appname`

    For namespace: if service came from `datname` (cloudsql), namespace
    follows the convention `<service>` (NOT the literal
    `monitoring`). Otherwise use `labels.namespace` when it's a real
    app namespace.
    """
    cls = (labels.get("class") or "").lower()
    grp = (labels.get("group") or "").lower()
    raw_ns = (labels.get("namespace") or "").lower() or None

    service: str | None = None
    ns_override: str | None = None

    # 1. Istio cross-service alerts.
    if labels.get("destination_workload"):
        service = labels["destination_workload"]
    # 2. HTTP probe / saas-outage. `app_instance` is the probed app.
    elif cls == "saas-outage" and labels.get("app_instance"):
        service = labels["app_instance"]
    # 3. CloudSQL: `datname` is the database name AND the owning service.
    elif cls == "cloudsql-alerts" and labels.get("datname"):
        service = labels["datname"]
        ns_override = labels["datname"]  # convention: namespace==service for apps
    # 4. Per-container alerts (ES index size etc.) -- `container` carries the service.
    elif labels.get("alertrule_appname") in ("alerting-elasticsearch",) and labels.get("container"):
        service = labels["container"]
        ns_override = labels["container"]
    # 5. Namespace-scoped app alerts (kafka memory, eck disk, devops-exporter).
    elif raw_ns and raw_ns not in _INFRA_NAMESPACES:
        service = raw_ns
    # 6. Summary tag fallback `[<env>/<tier>/<ns>]`.
    elif _ns_from_summary(summary):
        ns_from_tag = _ns_from_summary(summary)
        if ns_from_tag not in _INFRA_NAMESPACES:
            service = ns_from_tag
    # 7. Last-resort labels.
    if not service:
        service = labels.get("alertrule_release") or labels.get("alertrule_appname")

    # Namespace mapping
    namespace: str | None
    if ns_override:
        namespace = ns_override
    elif raw_ns and raw_ns not in _INFRA_NAMESPACES:
        namespace = raw_ns
    elif service:
        namespace = service  # convention default
    else:
        namespace = _ns_from_summary(summary)
    return service, namespace


def _map_payload_to_enrichment(payload: dict, match_source: str, match_score: float | None = None) -> AlertEnrichment:
    """Map the raw Rootly /v1/alerts/<id> response into AlertEnrichment.

    Pulls primarily from `data.alerts[0].labels` + `.annotations` because
    those are the structured fields (the `description` blob is just a
    human-readable rendering of the same data).
    """
    data = (payload.get("data") or {})
    attrs = (data.get("attributes") or {})
    # Nested AlertManager-style alerts array.
    nested = (attrs.get("data") or {}).get("alerts") or []
    labels: dict = {}
    annotations: dict = {}
    if nested:
        first = nested[0] or {}
        labels = first.get("labels") or {}
        annotations = first.get("annotations") or {}

    summary = attrs.get("summary") or annotations.get("summary") or ""
    service, namespace = _derive_service_and_namespace(labels, summary)

    return AlertEnrichment(
        summary=summary,
        service=service,
        namespace=namespace,
        env=(labels.get("env") or "").lower() or None,
        source_workload=labels.get("source_workload"),
        destination_workload=labels.get("destination_workload"),
        promql_expression=annotations.get("expr"),
        runbook_url=annotations.get("runbook_url") or labels.get("alertrule_runbook_url"),
        severity=labels.get("severity") or annotations.get("severity"),
        status=attrs.get("status"),
        firing_value=annotations.get("value"),
        generator_url=attrs.get("external_url") or annotations.get("generatorURL"),
        short_id=attrs.get("short_id"),
        match_source=match_source,
        match_score=match_score,
    )


async def _get_alert_by_short_id(short_id: str) -> dict | None:
    """`GET /v1/alerts/<short_id>` -- Rootly accepts short_id as alias for
    UUID. Returns the raw JSON, or None on failure."""
    token = _resolve_token()
    if not token:
        _log.warning("rootly token missing -- alert resolution disabled")
        return None
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{ROOTLY_API_BASE}/alerts/{short_id}",
                headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            )
        if resp.status_code != 200:
            _log.info("rootly get_alert short_id=%s -> %d", short_id, resp.status_code)
            return None
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        _log.warning("rootly get_alert short_id=%s failed: %s", short_id, exc)
        return None


async def _list_recent_alerts(limit: int = 50) -> list[dict]:
    """List recent alerts, newest first. 60s in-memory cache so a quick
    retry doesn't hammer Rootly. Returns minimal `{short_id, summary,
    created_at}` entries -- full payload is fetched only on the matched id.
    """
    now = time.time()
    if (now - _recent_cache["ts"]) < _RECENT_CACHE_TTL_SEC and _recent_cache["items"]:
        return _recent_cache["items"]
    token = _resolve_token()
    if not token:
        return []
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{ROOTLY_API_BASE}/alerts",
                params={"page[size]": str(limit), "sort": "-created_at"},
                headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            )
        if resp.status_code != 200:
            _log.info("rootly list_alerts -> %d", resp.status_code)
            return []
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001
        _log.warning("rootly list_alerts failed: %s", exc)
        return []
    items = []
    for it in (payload.get("data") or []):
        attrs = it.get("attributes") or {}
        short = attrs.get("short_id")
        if not short:
            continue
        items.append({
            "short_id": short,
            "summary": attrs.get("summary") or "",
            "created_at": attrs.get("created_at") or "",
        })
    _recent_cache["ts"] = now
    _recent_cache["items"] = items
    return items


def _age_seconds(iso_ts: str, now: float) -> float:
    try:
        from datetime import datetime
        # Rootly returns `2026-05-13T07:32:44.347-07:00` -- fromisoformat OK in py3.11+.
        return max(0.0, now - datetime.fromisoformat(iso_ts).timestamp())
    except Exception:
        return float("inf")


async def resolve_alert(
    alert_text: str,
    *,
    recency_seconds: float = _DEFAULT_RECENCY_SEC,
    threshold: float = _DEFAULT_JACCARD_THRESHOLD,
) -> AlertEnrichment | None:
    """Hybrid resolver:
      1. URL-in-text -> fetch by short_id (no scoring needed).
      2. Free text -> list recent, Jaccard-score each, pick best >= threshold
         AND created < recency_seconds ago. Then fetch the matched short_id
         for the full payload.

    Returns None when nothing matches -- caller should fall back to the
    LLM extractor.
    """
    if not alert_text:
        return None

    # Path A: Rootly URL -> direct fetch.
    short_id = parse_rootly_short_id(alert_text)
    if short_id:
        payload = await _get_alert_by_short_id(short_id)
        if payload is None:
            return None
        enr = _map_payload_to_enrichment(payload, match_source="rootly:url")
        _log.info(
            "rootly URL resolved short_id=%s service=%s ns=%s env=%s runbook=%s",
            enr.short_id, enr.service, enr.namespace, enr.env, bool(enr.runbook_url),
        )
        return enr

    # Path B: title fuzzy-match against recent alerts.
    recent = await _list_recent_alerts(limit=50)
    if not recent:
        return None
    target_tokens = _normalize_title(alert_text)
    if not target_tokens:
        return None
    now = time.time()
    best: tuple[float, dict] | None = None
    for it in recent:
        if _age_seconds(it.get("created_at", ""), now) > recency_seconds:
            continue
        score = _jaccard(target_tokens, _normalize_title(it.get("summary", "")))
        if score < threshold:
            continue
        if best is None or score > best[0]:
            best = (score, it)
    if best is None:
        _log.info("rootly title match: no candidate >= %.2f within %.0fh", threshold, recency_seconds / 3600)
        return None

    matched_short = best[1]["short_id"]
    payload = await _get_alert_by_short_id(matched_short)
    if payload is None:
        return None
    enr = _map_payload_to_enrichment(
        payload,
        match_source=f"rootly:match:{matched_short}",
        match_score=best[0],
    )
    _log.info(
        "rootly title-match short_id=%s score=%.2f service=%s ns=%s env=%s",
        enr.short_id, best[0], enr.service, enr.namespace, enr.env,
    )
    return enr
