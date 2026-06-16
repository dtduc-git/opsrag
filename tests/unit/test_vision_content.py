import base64

from opsrag.llms.content import (
    ImagePart,
    build_user_content,
    default_vision_model,
    is_multimodal,
    is_vision_capable,
    to_anthropic_content,
    to_bedrock_content,
    to_openai_content,
)

PNG = b"\x89PNG\r\n\x1a\nFAKEBYTES"


def test_no_images_returns_plain_string_fast_path():
    assert build_user_content("hello", []) == "hello"
    assert is_multimodal("hello") is False


def test_with_images_returns_parts_list():
    content = build_user_content("look", [ImagePart(PNG, "image/png", "a.png")])
    assert is_multimodal(content) is True
    assert content[0] == {"type": "text", "text": "look"}
    assert content[1]["type"] == "image"
    assert content[1]["mime_type"] == "image/png"
    assert content[1]["data"] == PNG


def test_anthropic_converts_image_to_base64_block():
    content = build_user_content("q", [ImagePart(PNG, "image/png")])
    out = to_anthropic_content(content)
    assert out[0] == {"type": "text", "text": "q"}
    assert out[1]["type"] == "image"
    assert out[1]["source"]["type"] == "base64"
    assert out[1]["source"]["media_type"] == "image/png"
    assert out[1]["source"]["data"] == base64.b64encode(PNG).decode("ascii")


def test_anthropic_passthrough_string():
    assert to_anthropic_content("plain") == "plain"


def test_bedrock_wraps_string_in_text_block():
    assert to_bedrock_content("plain") == [{"text": "plain"}]


def test_bedrock_converts_image_to_format_and_raw_bytes():
    content = build_user_content("q", [ImagePart(PNG, "image/jpeg")])
    out = to_bedrock_content(content)
    assert out[0] == {"text": "q"}
    assert out[1]["image"]["format"] == "jpeg"
    assert out[1]["image"]["source"]["bytes"] == PNG  # raw bytes, NOT base64


def test_openai_converts_image_to_data_url():
    content = build_user_content("q", [ImagePart(PNG, "image/webp")])
    out = to_openai_content(content)
    assert out[0] == {"type": "text", "text": "q"}
    b64 = base64.b64encode(PNG).decode("ascii")
    assert out[1] == {
        "type": "image_url",
        "image_url": {"url": f"data:image/webp;base64,{b64}"},
    }


def test_vision_capable_known_models():
    assert is_vision_capable("anthropic", "claude-sonnet-4-20250514") is True
    assert is_vision_capable("bedrock", "anthropic.claude-sonnet-4-20250514-v1:0") is True
    assert is_vision_capable("vertex", "gemini-3-flash-preview") is True
    assert is_vision_capable("openai", "gpt-4o") is True


def test_vision_incapable_models():
    assert is_vision_capable("openai", "gpt-3.5-turbo") is False
    assert is_vision_capable("litellm", "mistral/mistral-small") is False
    assert is_vision_capable("anthropic", "") is False


def test_default_vision_model_per_provider():
    assert default_vision_model("vertex") == "gemini-3-flash-preview"
    assert "claude" in default_vision_model("anthropic")
    assert "claude" in default_vision_model("bedrock")
    assert default_vision_model("litellm") is None
