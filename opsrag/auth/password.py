"""Argon2id password hashing/verification (auth Tier 2, login mode).

Thin wrapper over ``pwdlib`` configured with the Argon2id hasher. We do
NOT use passlib (unmaintained on 3.12+); ``pwdlib`` is the maintained
successor and ships an Argon2id hasher backed by ``argon2-cffi``.

Design
------
* One process-wide :data:`_password_hash` instance. ``pwdlib`` is
  stateless and thread-safe (the underlying ``argon2-cffi`` hasher is),
  so a singleton avoids re-allocating the hasher per request.
* :func:`hash_password` returns a self-describing PHC string
  (``$argon2id$v=19$m=...,t=...,p=...$salt$hash``) -- the salt and all
  parameters are embedded, so verification needs only the stored string.
* :func:`verify_password` returns ``(ok, new_hash)``. ``new_hash`` is a
  non-``None`` upgraded hash when the stored hash used weaker parameters
  than the current policy (pwdlib's verify-and-upgrade); callers SHOULD
  persist it. ``ok`` is constant-time within argon2-cffi.
* :func:`needs_rehash` lets a caller proactively detect a stored hash
  that should be re-hashed on next successful login.

This module deliberately knows nothing about users, stores, or HTTP --
it is pure crypto so it is trivial to unit-test.
"""
from __future__ import annotations

from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher

# A single Argon2id hasher. pwdlib's PasswordHash takes an ordered tuple
# of hashers; the first is the *current* policy used for hashing, and any
# of them can verify (so adding e.g. a bcrypt hasher later enables
# verify-and-upgrade of legacy hashes without breaking existing logins).
#
# Argon2Hasher defaults (argon2-cffi RFC 9106 "second recommended"
# profile: t=3, m=64MiB, p=4) are a sensible interactive-login policy.
_password_hash = PasswordHash((Argon2Hasher(),))


def hash_password(password: str) -> str:
    """Hash ``password`` with Argon2id; return the PHC-format string.

    The returned string embeds the algorithm, parameters, and a random
    per-call salt -- store it verbatim. Raises ``ValueError`` on an empty
    password (refusing to mint a credential for the empty string is a
    cheap foot-gun guard; real password-policy lives at the API layer).
    """
    if not password:
        raise ValueError("refusing to hash an empty password")
    return _password_hash.hash(password)


def verify_password(
    password: str,
    stored_hash: str | None,
) -> tuple[bool, str | None]:
    """Verify ``password`` against ``stored_hash``.

    Returns ``(ok, new_hash)``:
      * ``ok`` -- True iff the password matches. A ``None``/empty stored
        hash (e.g. an SSO-only account with ``password_hash IS NULL``)
        always returns ``(False, None)`` -- such accounts cannot log in
        with a password.
      * ``new_hash`` -- a freshly-computed hash when the stored hash used
        weaker-than-current parameters (verify-and-upgrade); the caller
        SHOULD persist it. ``None`` when no upgrade is needed.

    Verification is constant-time within argon2-cffi for a given stored
    hash. A malformed stored hash is treated as a non-match rather than
    raising, so a corrupt row can never 500 the login path.
    """
    if not stored_hash:
        return (False, None)
    try:
        return _password_hash.verify_and_update(password, stored_hash)
    except Exception:
        # Malformed/unknown hash format -> treat as a failed verification
        # rather than leaking a 500. (pwdlib raises on an unparseable hash.)
        return (False, None)


def needs_rehash(stored_hash: str | None) -> bool:
    """True iff ``stored_hash`` should be re-hashed under current policy.

    pwdlib only surfaces the rehash decision through
    ``verify_and_update`` (which needs the plaintext), so the canonical
    upgrade path is :func:`verify_password` returning a non-``None``
    ``new_hash`` on a successful login. This helper is a best-effort
    plaintext-free check: it returns True only when the stored hash is
    parseable but not in current Argon2id PHC form (e.g. a legacy
    algorithm), and False otherwise. Treat it as advisory.
    """
    if not stored_hash:
        return False
    # An Argon2id PHC string from the current hasher starts with
    # "$argon2id$"; anything else parseable is a candidate for upgrade.
    return not stored_hash.startswith("$argon2id$")
