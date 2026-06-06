"""Closed enums for investigation + runbook classification.

These are the dimensions used both for:
  - tagging historical investigations (Phase 3 auto-tagger writes to
    Qdrant payload `tags.*`), and
  - filtering / boosting hand-authored runbooks (`issue_kind` column on
    the `opsrag_runbooks` table is one of `FAILURE_CLASSES`).

Closed enums (not arbitrary strings) so retrieval can do exact-match
filtering without fuzzy NLU. The LLM-side auto-tagger is constrained
to these enums by its system prompt.
"""
from __future__ import annotations

# -- Failure class -- pick ONE per investigation ---------------------
# What KIND of failure was this. Eight classes chosen to cover the
# common operational shapes; expand only when a real incident doesn't
# fit any existing class.
FAILURE_CLASSES = (
    "deploy_regression",      # code/config push broke it; rollback or fix-forward
    "dependency_outage",      # DB / cache / queue / external API went down
    "infra_change",           # DBA / k8s / network change (e.g. CloudSQL UPDATE op)
    "resource_exhaustion",    # OOM, CPU, disk, rate-limit, connection pool
    "config_change",          # helm values, env var, feature flag
    "external_vendor",        # Cloudflare, GCP, vendor API
    "data_quality",           # bad input data, migration, corrupt state
    "unknown_recovered",      # self-resolved, no RCA done (transient / flaky)
)

# -- Symptom class -- pick ONE OR MORE -------------------------------
# What did the user SEE. Used to retrieve "similar shape" past
# investigations even when failure_class differs.
SYMPTOM_CLASSES = (
    "outage_full",            # service totally unreachable
    "outage_partial",         # some endpoints / regions / users affected
    "degraded_latency",       # slow but working
    "error_rate_spike",       # 5xx rate elevated
    "restart_loop",           # CrashLoopBackOff / repeated SIGKILL
    "silent_failure",         # no error reported; output wrong / missing
)

# -- Resolution class -- pick ONE ------------------------------------
# How was it FIXED. Useful when "I've seen this before -- what did we do?"
RESOLUTION_CLASSES = (
    "rollback",               # reverted the change
    "scale",                  # added capacity (HPA, manual scale, vertical bump)
    "restart",                # rolling restart / pod kill
    "config_revert",          # changed a config value back
    "dba_action",             # DBA intervention (kill query, restart instance, etc.)
    "vendor_action",          # waited for / escalated to external vendor
    "self_resolved",          # came back without intervention
    "no_action",              # known noise / suppressed
    "still_open",             # not yet resolved at investigation close
)

# -- Severity -------------------------------------------------------
SEVERITIES = ("SEV1", "SEV2", "SEV3", "SEV4")


def is_valid_failure_class(value: str | None) -> bool:
    return value is not None and value in FAILURE_CLASSES


def is_valid_symptom_class(value: str | None) -> bool:
    return value is not None and value in SYMPTOM_CLASSES


def is_valid_resolution_class(value: str | None) -> bool:
    return value is not None and value in RESOLUTION_CLASSES


def is_valid_severity(value: str | None) -> bool:
    return value is not None and value in SEVERITIES


# -- Recurrence marker -- sentinel constants -------------------------
# Format: "novel" OR "repeat:<N>" where N is the count (e.g. "repeat:3"
# = third time we've seen this shape). Computed by the auto-tagger
# based on Lane B historical retrieval at tag-write time.
RECURRENCE_NOVEL = "novel"


def make_recurrence_marker(repeat_count: int) -> str:
    if repeat_count <= 0:
        return RECURRENCE_NOVEL
    return f"repeat:{repeat_count}"
