# Postmortem - Acme Notes login outage (2026-03-20)

**Severity:** Sev-1
**Duration:** 33 minutes
**Author:** Acme Notes platform team

## Summary

New logins to Acme Notes failed for 33 minutes after the OIDC signing
certificate used by the identity provider expired and was not rotated in time.

## Impact

- Users with an active session were unaffected.
- 100% of new logins failed during the window.
- Support saw a spike in "cannot sign in" reports.

## Timeline

- 02:00 - The IdP signing certificate expires.
- 09:15 - Morning login traffic ramps; token validation starts failing.
- 09:21 - Login-failure alert pages on-call.
- 09:38 - On-call traces the failure to an expired signing key in the JWKS.
- 09:48 - IdP rotates the certificate; new keys published to the JWKS endpoint.
- 09:54 - Logins recover after the backend refreshes its JWKS cache.

## Detection

A synthetic login probe caught the failure, but only after real users were
already affected, because the probe ran every 10 minutes.

## Root cause

The OIDC signing certificate had a fixed expiry and no automated rotation. The
backend correctly rejected tokens signed by the expired key, but no alert
warned that the certificate was near expiry.

## Mitigation

The IdP team rotated the certificate and published fresh keys to the JWKS
endpoint. The backend picked them up automatically once its JWKS cache
(`jwks_cache_seconds`) expired.

## Action items

- Alert 14 days before any signing certificate expires.
- Automate certificate rotation on the identity provider.
- Shorten the synthetic login probe interval to 2 minutes.

## Lessons learned

Token validation behaved correctly; the gap was operational - no expiry alert
and no automated rotation.
