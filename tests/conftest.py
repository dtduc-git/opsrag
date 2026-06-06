"""Shared pytest fixtures for the opsrag test suite."""
from __future__ import annotations

from pathlib import Path

import pytest

# Repo root = two levels up from this file (tests/conftest.py -> repo/).
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def config_path(repo_root: Path) -> Path:
    """Absolute path to the shipped default config.yaml."""
    path = repo_root / "config.yaml"
    assert path.is_file(), f"shipped config.yaml not found at {path}"
    return path


# ---------------------------------------------------------------------------
# API contract-test fixtures.
#
# These build the FastAPI app WITHOUT running its lifespan, so no providers,
# database, vector store, or live IdP are required. The OIDC verifier is
# replaced with an offline stub so authenticated paths are reachable.
# ---------------------------------------------------------------------------
VALID_TEST_TOKEN = "valid-test-token"


class StubOIDCVerifier:
    """Offline stand-in for OIDCVerifier. Accepts exactly VALID_TEST_TOKEN
    and rejects everything else, with no network / JWKS involved."""

    def __init__(self, sub: str = "user-abc", email: str = "evaluator@example.com") -> None:
        self._sub = sub
        self._email = email

    def verify(self, token: str) -> dict:
        if token == VALID_TEST_TOKEN:
            return {
                "sub": self._sub,
                "email": self._email,
                "iss": "https://idp.test",
                "aud": "opsrag",
                "exp": 9999999999,
            }
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail="invalid token")

    def verify_to_user(self, token: str):
        from opsrag.auth.oidc import CurrentUser

        claims = self.verify(token)
        return CurrentUser(
            sub=claims["sub"],
            email=claims.get("email"),
            name=None,
            picture_url=None,
            groups=(),
            is_anonymous=False,
        )


@pytest.fixture
def api_app():
    """A freshly constructed app (lifespan not run)."""
    import opsrag.api.server as srv

    return srv.create_app()


@pytest.fixture
def stub_app(api_app):
    """App with the OIDC verifier swapped for the offline stub."""
    api_app.state.oidc_verifier = StubOIDCVerifier()
    return api_app


@pytest.fixture
def auth_headers() -> dict:
    return {"Authorization": f"Bearer {VALID_TEST_TOKEN}"}
