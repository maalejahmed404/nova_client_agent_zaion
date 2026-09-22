import sys

import pytest

from neova.config import get_settings


def _patch_dir(monkeypatch, name, path):
    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("neova") and hasattr(module, name):
            monkeypatch.setattr(module, name, path)


@pytest.fixture(autouse=True)
def mock_settings_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake_api_key")
    monkeypatch.setenv("CHAT_MODEL", "fake_chat_model")
    monkeypatch.setenv("CHAT_MODEL_FALLBACK", "fake_chat_model_fallback")
    monkeypatch.setenv("EMBEDDING_MODEL", "fake_embedding_model")
    monkeypatch.setenv("VISION_MODEL", "fake_vision_model")
    monkeypatch.setenv("REFERENCE_NOW", "2026-08-25T10:30:00")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    _patch_dir(monkeypatch, "CACHE_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def logs_dir(tmp_path, monkeypatch):
    _patch_dir(monkeypatch, "LOGS_DIR", tmp_path)
    return tmp_path