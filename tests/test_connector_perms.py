"""Unit tests for the per-connector permission resolver.

The resolver (``opsrag.auth.connector_perms.resolve_allowed_connectors``) is a
pure function: given a user's roles, the config ``role_connectors`` map, the
set of ``restricted`` connectors, the ``enabled`` connectors, and the user's
per-user allow/deny overrides, it returns the connectors that user may use.

Policy (design):
  * default-allow: a NON-restricted enabled connector is usable by anyone.
  * restricted connectors need an explicit grant (role_connectors OR per-user
    allow).
  * the ``admin`` role (and a role granting ``"*"``) implies every enabled
    connector.
  * per-user allow adds (even a restricted one); per-user deny wins over
    everything.
"""
from __future__ import annotations

from opsrag.auth.connector_perms import resolve_allowed_connectors

ENABLED = ("datadog", "gitlab", "elasticsearch", "gcp")


def _resolve(**kw):
    base = dict(
        roles=(),
        role_connectors=None,
        restricted=(),
        enabled_connectors=ENABLED,
        user_allow=(),
        user_deny=(),
    )
    base.update(kw)
    return resolve_allowed_connectors(**base)


def test_default_allow_when_nothing_restricted():
    # No restricted connectors -> every enabled connector is usable.
    assert _resolve(roles=("member_investigate",)) == frozenset(ENABLED)


def test_restricted_connector_hidden_without_grant():
    # datadog restricted, user has no grant -> datadog excluded, rest allowed.
    out = _resolve(roles=("member_investigate",), restricted=("datadog",))
    assert "datadog" not in out
    assert out == frozenset(("gitlab", "elasticsearch", "gcp"))


def test_restricted_connector_granted_via_role():
    out = _resolve(
        roles=("finance",),
        restricted=("datadog",),
        role_connectors={"finance": ["datadog"]},
    )
    assert "datadog" in out


def test_admin_role_gets_everything_including_restricted():
    out = _resolve(roles=("admin",), restricted=("datadog", "gcp"))
    assert out == frozenset(ENABLED)


def test_wildcard_role_grant_gets_everything():
    out = _resolve(
        roles=("superuser",),
        restricted=("datadog", "gcp"),
        role_connectors={"superuser": ["*"]},
    )
    assert out == frozenset(ENABLED)


def test_per_user_allow_grants_a_restricted_connector():
    out = _resolve(
        roles=("member_investigate",),
        restricted=("datadog",),
        user_allow=("datadog",),
    )
    assert "datadog" in out


def test_per_user_deny_wins_over_default_allow():
    # gitlab is not restricted (default-allow) but explicitly denied.
    out = _resolve(roles=("member_investigate",), user_deny=("gitlab",))
    assert "gitlab" not in out


def test_per_user_deny_wins_over_role_grant():
    out = _resolve(
        roles=("finance",),
        restricted=("datadog",),
        role_connectors={"finance": ["datadog"]},
        user_deny=("datadog",),
    )
    assert "datadog" not in out


def test_per_user_deny_wins_over_admin():
    # Even an admin can be denied a specific connector by an explicit deny.
    out = _resolve(roles=("admin",), user_deny=("datadog",))
    assert "datadog" not in out
    assert "gitlab" in out


def test_allow_and_deny_only_reference_enabled_connectors():
    # A grant/deny for a connector that isn't enabled is a no-op.
    out = _resolve(
        roles=("member_investigate",),
        user_allow=("pagerduty",),  # not in ENABLED
    )
    assert out == frozenset(ENABLED)
    assert "pagerduty" not in out


def test_result_never_exceeds_enabled_set():
    out = _resolve(roles=("admin",), user_allow=("splunk", "sentry"))
    assert out <= frozenset(ENABLED)


def test_empty_when_no_connectors_enabled():
    assert _resolve(roles=("admin",), enabled_connectors=()) == frozenset()
