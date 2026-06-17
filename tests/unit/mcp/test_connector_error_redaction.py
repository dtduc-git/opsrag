"""Credential-leak guard for connector error bodies (TRACK redact / M6).

The rootly / gitlab / pagerduty / datadog connectors used to raise/return
upstream error bodies verbatim to the LLM. Rootly post-mortems and CI job
traces in particular routinely contain pasted secrets, so a failing
request could echo a live token straight into the model context.

These tests assert the fix is applied AT THE SOURCE -- each connector's
`*MCPError.__init__` must scrub the body before storing it on `.body` and
before building the exception message. github.py / loki.py / sentry.py are
the correct reference impls (they already scrub); the four error classes
below must behave identically.
"""
from __future__ import annotations

import pytest

from opsrag.mcp.datadog import DatadogMCPError
from opsrag.mcp.gitlab import GitLabMCPError
from opsrag.mcp.pagerduty import PagerDutyMCPError
from opsrag.mcp.rootly import RootlyMCPError

# A planted secret of each major shape, plus a marker we keep so we can
# prove the non-secret text survives (redaction must scrub tokens, not
# nuke the whole body -- the agent still needs the surrounding context).
_GLPAT = "glpat-ABCDEFGHIJKLMNOPQRST"          # GitLab PAT
_ROOTLY = "rootly_" + "a" * 40                    # Rootly API token
_GHP = "ghp_" + "A" * 36                          # GitHub token
_AKIA = "AKIAIOSFODNN7EXAMPLE"                     # AWS access key id
_SLACK = "xoxb-" + "1" * 30                        # Slack bot token
_MARKER = "deployment failed at step build"


def _all_secrets() -> str:
    return (
        f"{_MARKER} token={_GLPAT} key={_ROOTLY} "
        f"gh={_GHP} aws={_AKIA} slack={_SLACK}"
    )


# (error class, default-tool-tag) for every connector under remediation.
_ERROR_CLASSES = [
    (RootlyMCPError, "rootly"),
    (GitLabMCPError, "gitlab"),
    (PagerDutyMCPError, "pagerduty"),
    (DatadogMCPError, "datadog"),
]

_SECRET_TOKENS = [_GLPAT, _ROOTLY, _GHP, _AKIA, _SLACK]


@pytest.mark.parametrize("error_cls,tag", _ERROR_CLASSES)
def test_planted_secrets_scrubbed_from_body_and_str(error_cls, tag) -> None:
    """Every planted token must be gone from BOTH .body and str(exc)."""
    body = _all_secrets()
    exc = error_cls(403, body, tool=f"{tag}_tool")

    for tok in _SECRET_TOKENS:
        assert tok not in exc.body, f"{error_cls.__name__}.body leaked {tok!r}"
        assert tok not in str(exc), f"str({error_cls.__name__}) leaked {tok!r}"

    # Redaction markers replace the secrets, and surrounding context stays.
    assert "[REDACTED:" in exc.body
    assert _MARKER in exc.body
    # Status + tool tag are still surfaced for the agent.
    assert exc.status == 403
    assert f"{tag}_tool" in str(exc)


@pytest.mark.parametrize("error_cls,tag", _ERROR_CLASSES)
def test_message_built_from_redacted_body(error_cls, tag) -> None:
    """The exception MESSAGE must derive from the already-redacted body --
    a secret in the first 300 chars must not survive in str(exc)."""
    body = f"FATAL {_GLPAT} {_ROOTLY} " + "x" * 5000
    exc = error_cls(500, body, tool=tag)
    assert _GLPAT not in str(exc)
    assert _ROOTLY not in str(exc)
    assert _GLPAT not in exc.body
    assert _ROOTLY not in exc.body


@pytest.mark.parametrize("error_cls,tag", _ERROR_CLASSES)
def test_none_body_does_not_crash(error_cls, tag) -> None:
    """A None body (some httpx paths pass through) must not raise."""
    exc = error_cls(404, None, tool=tag)  # type: ignore[arg-type]
    assert exc.body == ""
    assert tag in str(exc)


def test_gitlab_grep_job_trace_callsite_is_safe() -> None:
    """The whole point of redacting at the source: every `exc.body`
    call-site (e.g. gitlab `_h_grep_job_trace`) becomes safe for free.
    Simulate that call-site slicing `exc.body[:200]` into a tool result."""
    leaky_trace = f"$ git clone https://oauth2:{_GLPAT}@gitlab/x.git\nfatal: auth"
    exc = GitLabMCPError(401, leaky_trace, tool="gitlab_grep_job_trace")
    # This mirrors `f"unauthorized fetching trace: {exc.body[:200]}"`.
    surfaced = f"unauthorized fetching trace: {exc.body[:200]}"
    assert _GLPAT not in surfaced
    assert "[REDACTED:gitlab_token]" in surfaced
