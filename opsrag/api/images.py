"""Decode + validate web-attached images into ephemeral ImageParts."""
from __future__ import annotations

import base64
import binascii

from opsrag.api.models import ImageInput
from opsrag.config import VisionConfig
from opsrag.llms.content import ImagePart


class ImageValidationError(ValueError):
    """Raised when an attached image violates VisionConfig limits."""


def decode_images(images: list[ImageInput] | None, vision: VisionConfig) -> list[ImagePart]:
    if not images:
        return []
    if not vision.enabled:
        raise ImageValidationError("Image input is disabled on this deployment.")
    if len(images) > vision.max_images:
        raise ImageValidationError(
            f"Too many images: {len(images)} > max {vision.max_images}."
        )
    out: list[ImagePart] = []
    for i, img in enumerate(images):
        if img.mime_type not in vision.allowed_mime:
            raise ImageValidationError(f"Unsupported image type: {img.mime_type}.")
        try:
            raw = base64.b64decode(img.data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ImageValidationError(f"Image {i} is not valid base64.") from exc
        if len(raw) > vision.max_bytes:
            raise ImageValidationError(
                f"Image {i} too large: {len(raw)} bytes > max {vision.max_bytes}."
            )
        out.append(ImagePart(data=raw, mime_type=img.mime_type, name=f"image-{i}"))
    return out
