import base64

import pytest
from pydantic import ValidationError

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
