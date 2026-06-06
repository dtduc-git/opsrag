"""Runbook generator -- Pro LLM call that converts ONE specific
investigation into a REUSABLE runbook draft in the standard template format.

Triggered when the user clicks "Save as runbook" on a successful
investigation. The Pro tier is intentional: this is a generalization
task (extract the CLASS of incidents from a SPECIFIC incident), not
a summarization task. Flash routinely writes "what happened TODAY"
instead of "what to do NEXT TIME".

The output goes straight into the markdown textarea for the user to
review + edit + Save (creates an `opsrag_runbooks` row with
source='auto', source_investigation_id=<id>).

Template structure:
  # What's Happening?
  ## Description
  ## Impact (Who/What is Affected + Severity)
  # Why?
  ## Possible Causes
  # How to Troubleshoot / Solve
  # Related Links
  # Notifications
"""
from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger("opsrag.runbooks.generator")


_SYSTEM_RUNBOOK_GENERATOR = """\
You convert ONE specific SRE investigation into a REUSABLE runbook covering the CLASS of incidents like it. You are NOT writing a postmortem of THIS incident; you are writing a generic playbook for the NEXT engineer who hits the same SHAPE of incident, possibly six months from now.

OUTPUT FORMAT -- strict markdown, no preamble, no fences around the whole document. Use the exact section headers below. Empty/unknown sections write "_TBD -- fill before publishing_".

# What's Happening?

## Description
- Bullet list: 2-4 generic symptoms (NOT specific timestamps from this incident)
- Bullet list: how this CLASS of failure is typically detected (alerts, dashboards, user reports)

> **Example:** API latency increased from <baseline> to <degraded> at <time> UTC (Datadog alert `<metric>`).

## Impact

### Who/What is Affected
- Users, systems, or regions impacted (in GENERIC terms -- "EU users" not "Bob the user")
- % of traffic or # of users affected (typical range, not the specific number this time)
- Business effect (e.g. checkout unavailable)

> **Example:** ~40% of EU users can't log in (`/auth` endpoints return 500s).

### Severity
- SEV1 -- Full outage
- SEV2 -- Major degradation
- SEV3 -- Partial failure
- SEV4 -- Minor / alert only

# Why?

## Possible Causes
List 3-6 GENERIC causes that could produce this symptom shape. The cause confirmed by THIS investigation should be ONE of them, but the runbook covers ALL likely causes -- that's the whole point of generalization.

- Recent deploy or config change
- Dependency outage (DB, cache, broker)
- Resource exhaustion (CPU, memory, rate limits)
- <add domain-specific causes from the investigation>

# How to Troubleshoot / Solve

Numbered steps, 5-8 entries. Each step a short verb-phrase + a 1-line clarification. Cite SPECIFIC commands the engineer should run (`kubectl`, `gcloud`, MCP tool name, etc.).

1. **Confirm the issue** -- check dashboards and logs to verify scope and symptoms.
2. **Check recent changes** -- identify any deploys or config updates; roll back if needed.
   - `gitlab_list_deployments(project="<group>/<service>")`
3. **Validate dependencies** -- ensure databases, APIs, and queues are healthy.
   - `cloudsql_list_operations(instance="<db_instance>")`
4. **Mitigate** -- restart services, clear caches, or scale resources as applicable.
5. **Verify recovery** -- monitor metrics, test endpoints, and confirm normal operation.

# Related Links

- **Dashboards:** <Datadog dashboard URL OR `_TBD_`>
- **Alerts:** <Prometheus / Datadog alert names>
- **Logs:** <Kibana data-view or shortcut>
- **Postmortem:** <Confluence URL when one exists, else `_TBD_`>
- **Incident:** <Rootly URL when one exists, else `_TBD_`>

# Notifications

- **Slack:** `#oncall`, `#infrastructure-alerts`
- **PagerDuty:** <service name OR `_TBD_`>
- **Incident lead:** `@username` (current oncall)
- **Stakeholder updates:** `#status-updates`

---

RULES -- applied to EVERY runbook you write:

1. **Generalize.** If THIS incident was a CloudSQL DBA update on `<db_instance>`, the runbook section "Possible Causes" should list "Database infrastructure change (DBA maintenance, CloudSQL UPDATE op)" as a generic cause -- not "DBA updated `<db_instance>` at `<incident_time>`". Specific facts go in the postmortem; the runbook is the playbook for NEXT TIME.

2. **Use OpsRAG MCP tool names where they're the right answer.** When step 2 says "check recent deploys", cite `gitlab_list_deployments(project="<group>/<service>")` -- that's the actual tool. When step 3 says "check DB ops", cite `cloudsql_list_operations(instance="<db_instance>")`. Concrete tool names beat abstract advice.

3. **Fill every section.** Sections with no signal: write `_TBD -- fill before publishing_`. NEVER leave a section blank or invent details. The user reviews and fills TBDs before publishing.

4. **No specific timestamps, user names, or one-off IDs.** Replace a concrete pipeline number with `<pipeline_id>`, a concrete pod name with `<pod_name>`, a concrete date-time with `<incident_time>`.

5. **Severity is a TIER not a NUMBER.** Just say "SEV1-2 (typically)" -- don't pick a single severity number from this one incident.

6. **Output markdown only.** No JSON wrapper, no preamble explaining what you're about to do, no fences around the whole doc. Just the markdown that goes into the editor.
"""


async def generate_runbook_draft(
    *,
    llm: Any,
    investigation_question: str,
    investigation_answer: str,
    tool_call_audit: list[dict] | None = None,
    incident_target: str | None = None,
    timeout_s: float = 60.0,
) -> str | None:
    """One Pro call. Returns the markdown body string (no fences), or
    None on LLM error / empty response.

    The caller writes the returned markdown into a textarea pre-filled
    on the New Runbook page. The user reviews + edits + saves.

    `incident_target` is the service the investigation was about
    (extracted by triage / hypothesizer). Threaded into the prompt
    so the runbook gets pre-filled with the service name where
    applicable.
    """
    if llm is None:
        return None

    audit_lines = []
    for a in (tool_call_audit or [])[:50]:
        n = a.get("name") if isinstance(a, dict) else None
        if not n:
            continue
        err = a.get("error")
        audit_lines.append(f"- {n}{' [ERROR]' if err else ''}")
    audit_block = "\n".join(audit_lines) or "(no tools fired)"

    user_msg = (
        f"PRIMARY SERVICE: {incident_target or '(unspecified)'}\n\n"
        f"INVESTIGATION QUESTION:\n{(investigation_question or '')[:2000]}\n\n"
        f"INVESTIGATION CONCLUSION:\n{(investigation_answer or '')[:12000]}\n\n"
        f"TOOLS THAT FIRED ({len(tool_call_audit or [])}):\n{audit_block}\n\n"
        f"Now write the reusable runbook draft per the template + rules in your system prompt. Output the full markdown body and NOTHING else."
    )

    import asyncio
    try:
        resp = await asyncio.wait_for(
            llm.generate(
                messages=[{"role": "user", "content": user_msg}],
                system_prompt=_SYSTEM_RUNBOOK_GENERATOR,
                temperature=0.2,
                # Runbooks can be long -- 8K covers a thorough draft
                # without truncating mid-step. 2.5 Pro handles this cheaply.
                max_tokens=8192,
                purpose="runbook_generator",
            ),
            timeout=timeout_s,
        )
        content = (resp.content or "").strip()
    except TimeoutError:
        _log.warning("runbook generator timeout >%.0fs", timeout_s)
        return None
    except Exception as exc:
        _log.warning("runbook generator LLM error: %s", exc)
        return None

    if not content:
        return None

    # Strip any fences the model might have wrapped the doc in despite
    # the system prompt forbidding it.
    if content.startswith("```"):
        # Remove leading ```...\n and trailing ```
        first_newline = content.find("\n")
        if first_newline > 0:
            content = content[first_newline + 1:]
        if content.endswith("```"):
            content = content[:-3].rstrip()

    _log.info(
        "runbook draft generated: %d chars (target=%s)",
        len(content), incident_target,
    )
    return content
