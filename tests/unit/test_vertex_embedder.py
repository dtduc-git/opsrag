"""Unit tests for the Vertex embedder's retry classifier (#14a).

The retry predicate ``_is_retryable`` previously relied SOLELY on fragile
string matching (``"429" in str(exc)``), which silently stops retrying if the
provider reworids or localizes an error. The fix prefers typed
``google.api_core.exceptions`` (ResourceExhausted / ServiceUnavailable /
DeadlineExceeded / InternalServerError / TooManyRequests) and keeps the string
match only as a fallback.

``opsrag.embedders.vertex`` imports ``vertexai`` at module top (the optional
``vertex`` extra). To keep this test runnable on the bare install we inject a
minimal fake ``vertexai`` AND a fake ``google.api_core.exceptions`` so the
typed-exception path is exercised without the real SDKs.
"""
from __future__ import annotations

import importlib
import sys
import types

import pytest


@pytest.fixture
def vertex_mod(monkeypatch):
    """Import opsrag.embedders.vertex with fake ``vertexai`` and a fake
    ``google.api_core.exceptions`` providing typed retryable exceptions."""
    # ---- fake vertexai (satisfies the module-top import) ----
    fake_vertexai = types.ModuleType("vertexai")
    fake_vertexai.init = lambda **kwargs: None
    fake_lm = types.ModuleType("vertexai.language_models")
    fake_lm.TextEmbeddingInput = type("TextEmbeddingInput", (), {})
    fake_lm.TextEmbeddingModel = type("TextEmbeddingModel", (), {})
    fake_vertexai.language_models = fake_lm
    monkeypatch.setitem(sys.modules, "vertexai", fake_vertexai)
    monkeypatch.setitem(sys.modules, "vertexai.language_models", fake_lm)

    # ---- fake google.api_core.exceptions (typed exceptions) ----
    google_pkg = types.ModuleType("google")
    api_core_pkg = types.ModuleType("google.api_core")
    gexc = types.ModuleType("google.api_core.exceptions")

    class _GoogleAPIError(Exception):
        pass

    class ResourceExhausted(_GoogleAPIError):
        pass

    class TooManyRequests(_GoogleAPIError):
        pass

    class ServiceUnavailable(_GoogleAPIError):
        pass

    class DeadlineExceeded(_GoogleAPIError):
        pass

    class InternalServerError(_GoogleAPIError):
        pass

    class NotFound(_GoogleAPIError):  # a NON-retryable typed exception
        pass

    for cls in (
        ResourceExhausted, TooManyRequests, ServiceUnavailable,
        DeadlineExceeded, InternalServerError, NotFound,
    ):
        setattr(gexc, cls.__name__, cls)
    api_core_pkg.exceptions = gexc
    google_pkg.api_core = api_core_pkg
    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.api_core", api_core_pkg)
    monkeypatch.setitem(sys.modules, "google.api_core.exceptions", gexc)

    # Fresh import so the module binds against our fakes.
    sys.modules.pop("opsrag.embedders.vertex", None)
    mod = importlib.import_module("opsrag.embedders.vertex")
    yield mod, gexc
    sys.modules.pop("opsrag.embedders.vertex", None)


def test_typed_retryable_exceptions_are_retried(vertex_mod):
    mod, gexc = vertex_mod
    # A bare message with none of the magic substrings -- only the TYPE marks
    # it retryable. String matching alone would have returned False here.
    for cls in (
        gexc.ResourceExhausted,
        gexc.TooManyRequests,
        gexc.ServiceUnavailable,
        gexc.DeadlineExceeded,
        gexc.InternalServerError,
    ):
        exc = cls("the operation could not be completed")
        assert mod._is_retryable(exc) is True, f"{cls.__name__} should be retryable"


def test_typed_non_retryable_exception_is_not_retried(vertex_mod):
    mod, gexc = vertex_mod
    # NotFound is a typed google error but NOT in the retryable set, and its
    # message carries none of the fallback substrings.
    exc = gexc.NotFound("model could not be located")
    assert mod._is_retryable(exc) is False


def test_string_fallback_still_classifies_untyped_errors(vertex_mod):
    mod, _ = vertex_mod
    # A plain Exception (untyped) carrying a 429/503 signal still retries via
    # the preserved string-match fallback.
    assert mod._is_retryable(Exception("HTTP 429: quota exceeded")) is True
    assert mod._is_retryable(Exception("503 Service Unavailable")) is True
    # And a genuinely benign error is not retried.
    assert mod._is_retryable(Exception("invalid argument: bad task_type")) is False


def test_retryable_types_empty_without_google(vertex_mod, monkeypatch):
    """If google.api_core can't be imported, _retryable_exc_types returns ()
    and the classifier degrades to pure string matching (no crash)."""
    mod, _ = vertex_mod

    import builtins
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "google.api_core" or name.startswith("google.api_core"):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    assert mod._retryable_exc_types() == ()
    # With no typed types resolvable, classification falls back to string
    # matching, which still works.
    assert mod._is_retryable(Exception("429 too many requests")) is True
    assert mod._is_retryable(Exception("invalid argument")) is False
