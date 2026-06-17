"""M4 (b): the spaCy NER discriminator is ON by default.

`cfg.qa_ner_guard` (QACacheConfig, default True) populates the module
default via `configure(...)`; the `OPSRAG_QA_NER_SPACY` env var still wins
when present. Graceful-degrade is preserved: with the guard enabled but
the spaCy model absent, `extract_entities` returns an empty set (it does
not raise), so a non-container install is unaffected.
"""
from __future__ import annotations

import importlib

import opsrag.qa_cache_ner as ner


def _reload_clean(monkeypatch):
    """Reload the module with a clean env so the module-level default is
    the pristine `_default_enabled = True` (no env override leaking in)."""
    monkeypatch.delenv("OPSRAG_QA_NER_SPACY", raising=False)
    return importlib.reload(ner)


def test_ner_guard_on_by_default(monkeypatch):
    m = _reload_clean(monkeypatch)
    # No env var set -> the config-derived default (True) governs.
    assert m.is_enabled() is True


def test_env_off_overrides_config_default(monkeypatch):
    m = _reload_clean(monkeypatch)
    monkeypatch.setenv("OPSRAG_QA_NER_SPACY", "0")
    # Env wins -> off even though the config default is True.
    assert m.is_enabled() is False


def test_env_on_overrides_config_off(monkeypatch):
    m = _reload_clean(monkeypatch)
    m.configure(qa_ner_guard=False)  # config says off
    assert m.is_enabled() is False
    monkeypatch.setenv("OPSRAG_QA_NER_SPACY", "1")  # env says on -> wins
    assert m.is_enabled() is True


def test_configure_applies_cfg_value(monkeypatch):
    m = _reload_clean(monkeypatch)
    m.configure(qa_ner_guard=False)
    assert m.is_enabled() is False
    m.configure(qa_ner_guard=True)
    assert m.is_enabled() is True
    # None is a no-op (keeps the current default).
    m.configure(qa_ner_guard=None)
    assert m.is_enabled() is True


def test_graceful_degrade_when_model_absent(monkeypatch):
    """Guard ON but spaCy model not loadable -> empty set, never raises.

    Force the load to fail by stubbing `_load_nlp` to return None (the
    same path the real graceful-degrade takes when spacy/model is absent).
    """
    m = _reload_clean(monkeypatch)
    assert m.is_enabled() is True
    monkeypatch.setattr(m, "_load_nlp", lambda: None)
    out = m.extract_entities("the Asia region had ten brokers down on Tuesday")
    assert out == frozenset()
