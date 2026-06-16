from opsrag.config import OpsRAGConfig, VisionConfig


def test_vision_defaults():
    v = VisionConfig()
    assert v.enabled is True
    assert v.max_images == 4
    assert v.max_bytes == 5 * 1024 * 1024
    assert set(v.allowed_mime) == {"image/png", "image/jpeg", "image/gif", "image/webp"}
    assert v.model is None
    assert v.provider is None


def test_top_level_config_has_vision_section():
    cfg = OpsRAGConfig()
    assert isinstance(cfg.vision, VisionConfig)
