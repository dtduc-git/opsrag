"""Auto-tagger -- Flash LLM call that classifies a closed investigation
into the 6-dim tag schema, then writes the tags onto the existing
Qdrant payload via `InvestigationCache.record_tags`.

Invoked at investigation close-time (in graph.py post-generator). Also
exposed as a callable for backfill scripts (`python -m opsrag.runbooks
.tagger_backfill`).

Flash is the right tier here -- pure classification, no creative
synthesis. Vertex Gemini Flash returns structured JSON reliably with a
temperature=0 + constrained-output prompt.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from opsrag.runbooks.taxonomy import (
    FAILURE_CLASSES,
    RESOLUTION_CLASSES,
    SEVERITIES,
    SYMPTOM_CLASSES,
    make_recurrence_marker,
)

_log = logging.getLogger("opsrag.runbooks.tagger")


_SYSTEM_TAGGER = """\
You are the AUTO-TAGGER for closed SRE investigations. Given the user question + the synthesized answer + the audit log of tools that fired, classify the investigation into a strict 6-dimension tag schema.

OUTPUT FORMAT -- strict JSON only, no prose, no markdown fences:

{
  "service": "<primary affected service name, e.g. '<service>'. Use null when no specific service was the subject>",
  "dependencies": ["<list of dependencies named in the investigation: cloudsql, redis, kafka, rabbitmq, gcs, dns, cloudflare, etc.>"],
  "failure_class": "<EXACTLY ONE OF: deploy_regression | dependency_outage | infra_change | resource_exhaustion | config_change | external_vendor | data_quality | unknown_recovered>",
  "symptom_class": ["<ONE OR MORE OF: outage_full, outage_partial, degraded_latency, error_rate_spike, restart_loop, silent_failure>"],
  "resolution_class": "<EXACTLY ONE OF: rollback | scale | restart | config_revert | dba_action | vendor_action | self_resolved | no_action | still_open>",
  "severity": "<SEV1 | SEV2 | SEV3 | SEV4 -- derive from impact described in the answer>"
}

CLASSIFICATION RULES -- read carefully, mistakes here poison future retrieval:

1. **failure_class -- what CAUSED it (not what the symptom was):**
   - "deploy_regression" -> a code/config push broke it (look for `gitlab_list_deployments` showing a recent deploy timestamp matching the failure onset)
   - "dependency_outage" -> a downstream service (DB / cache / queue / vendor API) was unhealthy / unreachable
   - "infra_change" -> a non-code change to the infrastructure: DBA updated a SQL instance, k8s cluster change, network change.
   - "resource_exhaustion" -> OOM, CPU limit, disk full, connection pool exhausted, rate-limited
   - "config_change" -> helm values or env var or feature flag changed (NOT a code push)
   - "external_vendor" -> Cloudflare / GCP / SaaS vendor issue (verify by vendor status page mentioned)
   - "data_quality" -> bad input data, migration regression, corrupt state
   - "unknown_recovered" -> self-resolved before RCA, transient blip

2. **symptom_class -- what the user OBSERVED. Pick all that apply:**
   - "outage_full" if NOTHING worked (all pods crashlooping, 100% error rate)
   - "outage_partial" if SOME endpoints / regions / users were affected
   - "degraded_latency" if the service was slow but functional
   - "error_rate_spike" if 5xx rate climbed but it wasn't a full outage
   - "restart_loop" if pods were CrashLoopBackOff / repeated SIGKILL
   - "silent_failure" if no error was reported but output was wrong / missing

3. **resolution_class -- how was it ACTUALLY fixed:**
   - "self_resolved" when investigation closed with no manual fix (and the user wasn't told to do something)
   - "still_open" when the investigation describes the cause but the user hasn't acted yet
   - "rollback" / "restart" / "scale" / "config_revert" / "dba_action" / "vendor_action" -- clear actions named in the answer
   - "no_action" if known noise / suppressed alert

4. **severity -- based on impact statement in the answer:**
   - SEV1 = full outage, paged on-call
   - SEV2 = major degradation, customer-visible
   - SEV3 = partial / single-tenant
   - SEV4 = minor / alert-only / staging-only

5. **service** -- must be a real service: the SUBJECT of the failure, not a downstream symptom. If the question is about <service-a>'s login flow failing, the service is "<service-a>" (NOT "<service-b>" which merely surfaced the symptom).

6. **dependencies** -- list every dep named in the investigation. Examples: ["cloudsql"], ["cloudsql", "redis"], ["kafka"], []. Empty list is valid.

ANTI-PATTERNS (common tagging mistakes to avoid):
- Don't put a dependency NAME in failure_class -- a broker/queue name is not a class; "dependency_outage" with dependencies=["<broker>"] is correct
- Don't tag "self_resolved" if the answer just describes the cause -- the user might still be acting
- Don't downgrade severity. If the impact statement describes a customer-facing capability down for all users of an environment, that's SEV2 minimum even if it self-resolved
- If you cannot determine a field from the inputs, use null / [] -- NEVER guess
"""


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*\n|\n```\s*$", re.MULTILINE)


async def autotag_investigation(
    *,
    llm: Any,
    question: str,
    answer: str,
    tool_call_audit: list[dict] | None = None,
    repeat_count: int = 0,
    auto_tagged: bool = True,
    timeout_s: float = 15.0,
) -> dict | None:
    """One Flash call. Returns the 6-dim tag dict, or None on parse failure.

    `repeat_count` is the number of similar past investigations in the
    cache (from Lane B). 0 -> "novel"; N>0 -> "repeat:N". Computed by the
    caller (graph.py knows the Lane B result count from the same turn).

    `auto_tagged=True` marks this as machine-tagged in the payload, so
    the UI can show "auto" badges + let users override.
    """
    if llm is None:
        return None
    audit_lines = []
    for a in (tool_call_audit or [])[:30]:
        n = a.get("name") if isinstance(a, dict) else None
        if not n:
            continue
        err = a.get("error")
        audit_lines.append(f"- {n}{' [ERROR]' if err else ''}")
    audit_block = "\n".join(audit_lines) or "(no tools fired)"

    user_msg = (
        f"INVESTIGATION TO CLASSIFY:\n\n"
        f"QUESTION:\n{(question or '')[:2000]}\n\n"
        f"ANSWER:\n{(answer or '')[:8000]}\n\n"
        f"TOOLS FIRED ({len(tool_call_audit or [])}):\n{audit_block}\n\n"
        f"Output strict JSON per the 6-dim schema in the system prompt."
    )

    import asyncio
    try:
        resp = await asyncio.wait_for(
            llm.generate(
                messages=[{"role": "user", "content": user_msg}],
                system_prompt=_SYSTEM_TAGGER,
                temperature=0.0,
                max_tokens=512,
                purpose="autotagger",
            ),
            timeout=timeout_s,
        )
        content = (resp.content or "").strip()
    except TimeoutError:
        _log.warning("autotagger timeout >%.0fs -- no tags", timeout_s)
        return None
    except Exception as exc:
        _log.warning("autotagger LLM error: %s -- no tags", exc)
        return None

    # Strip ```json fences if model emits them.
    content = _JSON_FENCE_RE.sub("", content).strip()
    try:
        parsed = json.loads(content)
    except Exception as exc:
        _log.warning(
            "autotagger JSON parse failed (%s); content head: %r",
            exc, content[:200],
        )
        return None

    # Validate + coerce. Anything invalid -> field becomes None / [] so
    # downstream filters skip rather than crash.
    out: dict = {}
    out["service"] = _none_if_blank(parsed.get("service"))
    out["dependencies"] = _list_of_strings(parsed.get("dependencies"))

    fc = parsed.get("failure_class")
    out["failure_class"] = fc if fc in FAILURE_CLASSES else None

    sc = parsed.get("symptom_class") or []
    if isinstance(sc, str):
        sc = [sc]
    out["symptom_class"] = [s for s in sc if isinstance(s, str) and s in SYMPTOM_CLASSES]

    rc = parsed.get("resolution_class")
    out["resolution_class"] = rc if rc in RESOLUTION_CLASSES else None

    sev = parsed.get("severity")
    out["severity"] = sev if sev in SEVERITIES else None

    out["recurrence_marker"] = make_recurrence_marker(repeat_count)
    out["auto_tagged"] = bool(auto_tagged)
    out["tagged_at"] = time.time()

    _log.info(
        "autotagger: service=%s failure=%s symptoms=%s resolution=%s sev=%s rec=%s",
        out["service"], out["failure_class"], out["symptom_class"],
        out["resolution_class"], out["severity"], out["recurrence_marker"],
    )
    return out


def _none_if_blank(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _list_of_strings(v: Any) -> list[str]:
    if not isinstance(v, list):
        return []
    return [str(x).strip() for x in v if x and isinstance(x, (str, bytes))][:8]
