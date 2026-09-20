import os

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("CHAT_MODEL", "test/chat")
os.environ.setdefault("CHAT_MODEL_FALLBACK", "test/fallback")
os.environ.setdefault("JUDGE_MODEL", "test/judge")
os.environ.setdefault("EMBEDDING_MODEL", "test/embed")
os.environ.setdefault("VISION_MODEL", "test/vision")
os.environ.setdefault("REFERENCE_NOW", "2026-08-25T10:00:00+02:00")
os.environ.setdefault("API_CHAOS_RATE", "0")

import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_store(monkeypatch):
    """Every test writes its runtime state to a temp file, never to data/runtime_state.json."""
    from neova.api.store import store

    with tempfile.TemporaryDirectory() as root:
        monkeypatch.setattr(store, "runtime_path", Path(root) / "runtime_state.json")
        store.reset()
        yield


