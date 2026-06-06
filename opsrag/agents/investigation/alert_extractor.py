"""Auto-extract `(service, namespace, env)` from a free-form alert.

The investigation subgraph used to require three explicit hints from
the UI. With this module the user pastes only the alert text and we
derive the same three fields server-side -- first by regex (covers the
common alert templates) then by Flash LLM if regex didn't
yield a service.

Defaults applied at the end:
  - env       -> "prod" (operators investigate prod by reflex)
  - namespace -> service (matches a common convention: namespace
                          equals service slug for ~= 95% of services)
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

_log = logging.getLogger("opsrag.agents.investigation.alert_extractor")

# Recognised env tokens -- lowercased. Maps any alias to a canonical
# k8s cluster shorthand we use elsewhere (`prod`, `staging`, `shared`).
_ENV_ALIASES = {
    "prod": "prod", "production": "prod",
    "staging": "staging",
    "preprod": "preprod",
    "shared": "shared", "sandbox": "shared",
    "dev": "dev", "development": "dev",
}

# Prefix tag like `[P2][prod/k8s/kong]` -- the alert subject
# convention puts env/cluster-tier/namespace there.
_PREFIX_TAG_RE = re.compile(r"\[\s*([a-z]{2,12})\s*/\s*[a-z0-9_-]+\s*/\s*([a-z0-9_-]+)\s*\]", re.IGNORECASE)
# Fully-qualified k8s service hostname `<service>.<namespace>` or
# `<service>.<namespace>.svc.cluster.local`. Pulls service from the
# first segment and namespace from the second.
_FQDN_RE = re.compile(r"\b([a-z0-9][a-z0-9-]{1,40})\.([a-z0-9][a-z0-9-]{1,40})(?:\.svc(?:\.cluster\.local)?|\.fms-default-main-public|[:\s])", re.IGNORECASE)
# Explicit `namespace=foo` / `service=bar` / `env=staging` in any form.
_EXPLICIT_RE = re.compile(r"\b(service|namespace|ns|env|environment|cluster)\s*[:=]\s*([a-z0-9_-]+)", re.IGNORECASE)


@dataclass
class AlertContext:
    service: str | None
    namespace: str | None
    env: str | None
    source: str  # "heuristic" | "llm" | "default" | "mixed"


def _normalize_env(token: str | None) -> str | None:
    if not token:
        return None
    return _ENV_ALIASES.get(token.lower())


def extract_heuristic(alert_text: str) -> AlertContext:
    """Best-effort regex extraction. Returns context with whatever
    fields could be derived. Caller layers defaults / LLM fallback on
    top.
    """
    service: str | None = None
    namespace: str | None = None
    env: str | None = None

    # 1. Explicit key=value fragments -- highest precedence.
    for m in _EXPLICIT_RE.finditer(alert_text):
        key, val = m.group(1).lower(), m.group(2)
        if key in ("service",):
            service = service or val
        elif key in ("namespace", "ns"):
            namespace = namespace or val
        elif key in ("env", "environment", "cluster"):
            env = env or _normalize_env(val)

    # 2. Prefix tag `[<env>/<tier>/<namespace>]`.
    m = _PREFIX_TAG_RE.search(alert_text)
    if m:
        env = env or _normalize_env(m.group(1))
        namespace = namespace or m.group(2)

    # 3. Fully-qualified k8s hostname `service.namespace[.svc...]`.
    for m in _FQDN_RE.finditer(alert_text):
        candidate_svc, candidate_ns = m.group(1), m.group(2)
        # Skip if both halves are the same generic word (`localhost.localhost`).
        if candidate_svc == candidate_ns:
            continue
        if not service:
            service = candidate_svc
        if not namespace:
            namespace = candidate_ns
        break

    return AlertContext(service=service, namespace=namespace, env=env, source="heuristic")


def _llm_prompt(alert_text: str) -> str:
    return (
        "You extract structured fields from an SRE alert.\n"
        "Return JSON ONLY (no prose, no fences) with three keys: "
        "`service`, `namespace`, `env`.\n\n"
        "Rules:\n"
        "- `service`: the application/service the alert is ABOUT. "
        "If the alert mentions an upstream gateway forwarding to a "
        "downstream service (e.g. Kong -> fms), pick the DOWNSTREAM "
        "service that's failing. Lowercase, hyphenated slug.\n"
        "- `namespace`: kubernetes namespace, lowercase. If not "
        "obviously present, return null and the caller will default "
        "to the service name.\n"
        "- `env`: one of `prod`, `staging`, `shared`, `dev`. If absent, "
        "return null and the caller will default to `prod`.\n"
        "- Use null for unknown fields; do not guess wildly.\n\n"
        f"Alert: {alert_text!r}\n\nJSON:"
    )


_LLM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "service":   {"type": ["string", "null"]},
        "namespace": {"type": ["string", "null"]},
        "env":       {"type": ["string", "null"]},
    },
    "required": ["service", "namespace", "env"],
}


async def extract_llm(alert_text: str, llm: Any) -> AlertContext:
    """LLM-based extraction via Flash. Returns context with whatever
    Flash filled in. Caller layers defaults on top.
    """
    try:
        resp = await llm.generate(
            messages=[{"role": "user", "content": _llm_prompt(alert_text)}],
            system_prompt="You are a precise field extractor. Output JSON only.",
            temperature=0.0,
            max_tokens=200,
            purpose="alert_extract",
            response_schema=_LLM_SCHEMA,
        )
    except TypeError:
        # LLM provider doesn't support response_schema kw -- retry without it.
        try:
            resp = await llm.generate(
                messages=[{"role": "user", "content": _llm_prompt(alert_text)}],
                system_prompt="You are a precise field extractor. Output JSON only.",
                temperature=0.0,
                max_tokens=200,
                purpose="alert_extract",
            )
        except Exception as exc:
            _log.warning("alert_extract LLM error: %s", exc)
            return AlertContext(None, None, None, source="llm")
    except Exception as exc:
        _log.warning("alert_extract LLM error: %s", exc)
        return AlertContext(None, None, None, source="llm")

    text = (getattr(resp, "content", "") or "").strip()
    # Strip ``` fences if the model added them despite instructions.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except Exception as exc:
        _log.warning("alert_extract LLM JSON parse failed: %s text=%r", exc, text[:200])
        return AlertContext(None, None, None, source="llm")

    def _clean(v: Any) -> str | None:
        if not isinstance(v, str):
            return None
        s = v.strip().lower()
        if not s or s in ("null", "none", "unknown", "n/a"):
            return None
        return s

    return AlertContext(
        service=_clean(data.get("service")),
        namespace=_clean(data.get("namespace")),
        env=_normalize_env(_clean(data.get("env"))),
        source="llm",
    )


async def extract_alert_context(
    alert_text: str,
    *,
    llm: Any | None = None,
    explicit: dict | None = None,
) -> AlertContext:
    """Hybrid extraction: regex first, LLM fallback only when service
    couldn't be derived heuristically. Apply defaults at the end so
    the subgraph never sees nulls.

    `explicit` carries any user-provided overrides (still supported
    even though the UI removed the fields -- keeps the JSON API stable
    for direct API users).
    """
    explicit = explicit or {}
    h = extract_heuristic(alert_text)
    service = explicit.get("service") or h.service
    namespace = explicit.get("namespace") or h.namespace
    env = _normalize_env(explicit.get("env")) or h.env
    source = "heuristic"

    if not service and llm is not None:
        l = await extract_llm(alert_text, llm)
        service = service or l.service
        namespace = namespace or l.namespace
        env = env or l.env
        source = "mixed" if (h.service or h.namespace or h.env) else "llm"

    # Defaults -- user spec:
    #   env       -> prod
    #   namespace -> service slug (common convention)
    if not env:
        env = "prod"
        source = source if source != "heuristic" else "heuristic+default"
    if not namespace and service:
        namespace = service
        source = source if "default" in source else source + "+default"

    return AlertContext(service=service, namespace=namespace, env=env, source=source)
