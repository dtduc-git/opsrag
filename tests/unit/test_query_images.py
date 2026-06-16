import asyncio
import base64

import pytest

from opsrag.api.images import ImageValidationError, decode_images
from opsrag.api.models import ImageInput, QueryRequest
from opsrag.config import VisionConfig

PNG_B64 = base64.b64encode(b"\x89PNG\r\nfake").decode()


def test_query_request_accepts_images():
    req = QueryRequest(query="hi", images=[ImageInput(mime_type="image/png", data=PNG_B64)])
    assert req.images[0].mime_type == "image/png"


def test_query_request_images_optional():
    assert QueryRequest(query="hi").images is None


def test_decode_rejects_bad_mime():
    v = VisionConfig()
    with pytest.raises(ImageValidationError):
        decode_images([ImageInput(mime_type="application/pdf", data=PNG_B64)], v)


def test_decode_rejects_too_many():
    v = VisionConfig(max_images=1)
    imgs = [
        ImageInput(mime_type="image/png", data=PNG_B64),
        ImageInput(mime_type="image/png", data=PNG_B64),
    ]
    with pytest.raises(ImageValidationError):
        decode_images(imgs, v)


def test_decode_rejects_oversize():
    v = VisionConfig(max_bytes=4)
    with pytest.raises(ImageValidationError):
        decode_images([ImageInput(mime_type="image/png", data=PNG_B64)], v)


def test_decode_ok_returns_image_parts():
    v = VisionConfig()
    parts = decode_images([ImageInput(mime_type="image/png", data=PNG_B64)], v)
    assert parts[0].mime_type == "image/png"
    assert parts[0].data == b"\x89PNG\r\nfake"


# --- Fix 1: bare-image turn (empty query) must validate on the web path ----
def test_query_request_accepts_empty_query():
    """A bare-image turn posts query="" -- must NOT 422 at the model layer.
    The route handler substitutes a default prompt when images are present."""
    assert QueryRequest(query="").query == ""


def test_query_request_empty_query_with_image_valid():
    req = QueryRequest(query="", images=[ImageInput(mime_type="image/png", data=PNG_B64)])
    assert req.query == ""
    assert req.images[0].mime_type == "image/png"


def test_query_request_query_defaults_to_empty():
    """query is now optional and defaults to empty string."""
    req = QueryRequest(images=[ImageInput(mime_type="image/png", data=PNG_B64)])
    assert req.query == ""


# --- Fix 1: route handler substitutes the default prompt for a bare image ---
def _query_handler_state():
    """Minimal app.state + providers for driving routes.query directly."""
    state = type("_State", (), {})()
    state.config = None
    state.semantic_router = None
    state.agent_graph = object()
    state.session_store = None
    providers = type("_Prov", (), {})()
    providers.embedder = None
    providers.llm = None
    providers.session_store = None
    providers.vision_llm = None
    state.providers = providers
    state.qa_cache = None
    state.investigation_cache = None
    req = type("_Req", (), {})()
    req.app = type("_App", (), {"state": state})()
    return req


def _run_query_handler(monkeypatch, query, images):
    from opsrag.api import routes
    from opsrag.auth.scopes import Scope

    captured = {}

    async def _fake_qws(graph, *, query, **kw):  # noqa: ANN001
        captured["query"] = query
        captured["images"] = kw.get("images")
        return {"answer": "ok", "sources": [], "grounded": True,
                "thread_id": "t1", "session_resumable": True}

    monkeypatch.setattr(routes, "query_with_session", _fake_qws)

    user = type("_U", (), {})()
    user.oid = "u1"
    user.email = None
    user.name = None
    user.is_anonymous = False
    user.scopes = frozenset({Scope.CHAT})

    api_req = QueryRequest(query=query, images=images, stream=False)
    http_req = _query_handler_state()
    asyncio.run(routes.query(api_req, http_req, current_user=user))
    return captured


def test_handler_substitutes_prompt_for_bare_image(monkeypatch):
    captured = _run_query_handler(
        monkeypatch, query="", images=[ImageInput(mime_type="image/png", data=PNG_B64)]
    )
    assert captured["query"] == "Please analyze this image."
    assert captured["images"]  # the decoded image rode through


def test_handler_rejects_genuinely_empty_turn(monkeypatch):
    from opsrag.api import routes

    with pytest.raises(routes.HTTPException) as ei:
        _run_query_handler(monkeypatch, query="   ", images=None)
    assert ei.value.status_code == 400


def test_handler_keeps_real_query_unchanged(monkeypatch):
    captured = _run_query_handler(monkeypatch, query="why is kafka slow", images=None)
    assert captured["query"] == "why is kafka slow"
