import json
import tempfile
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
import tenacity
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from neova import llm
from neova.config import get_settings


@pytest.fixture
def embeddings_url() -> str:
    return f"{get_settings().openrouter_base_url.rstrip('/')}/embeddings"


@pytest.fixture(autouse=True)
def isolated_dirs(monkeypatch):
    with tempfile.TemporaryDirectory() as root:
        monkeypatch.setattr(llm, "CACHE_DIR", Path(root) / "cache")
        monkeypatch.setattr(llm, "LOGS_DIR", Path(root) / "logs")
        yield


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch):
    monkeypatch.setattr(llm._post_embeddings.retry, "wait", tenacity.wait_none())


def embedding_response(*vectors: list[float]) -> httpx.Response:
    data = [{"index": i, "embedding": vector} for i, vector in enumerate(vectors)]
    return httpx.Response(200, json={"data": data})


def test_embed_texts_retries_after_429(respx_mock, embeddings_url):
    route = respx_mock.post(embeddings_url).mock(
        side_effect=[httpx.Response(429), embedding_response([0.1, 0.2])]
    )

    assert llm.embed_texts(["bonjour"]) == [[0.1, 0.2]]
    assert route.call_count == 2


def test_embed_texts_uses_cache_on_second_call(respx_mock, embeddings_url):
    route = respx_mock.post(embeddings_url).mock(return_value=embedding_response([0.3, 0.4]))

    first = llm.embed_texts(["bonjour"])
    calls_after_first = route.call_count
    second = llm.embed_texts(["bonjour"])

    assert first == second == [[0.3, 0.4]]
    assert route.call_count == calls_after_first == 1


def test_embed_texts_query_instruction_is_sent_and_cached_separately(respx_mock, embeddings_url):
    route = respx_mock.post(embeddings_url).mock(
        side_effect=[embedding_response([1.0]), embedding_response([2.0])]
    )

    assert llm.embed_texts(["bonjour"]) == [[1.0]]
    assert llm.embed_texts(["bonjour"], instruction="trouver") == [[2.0]]

    sent = json.loads(route.calls.last.request.content)["input"]
    assert sent == ["Instruct: trouver\nQuery: bonjour"]


def test_usage_logger_writes_one_record_from_llm_output():
    logger = llm.UsageLogger()
    run_id = uuid4()
    result = LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="ok"))]],
        llm_output={
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model_name": "test/chat",
        },
    )

    logger.on_llm_start({}, ["hi"], run_id=run_id, invocation_params={"model": "test/chat"})
    logger.on_llm_end(result, run_id=run_id, tags=["triage"])

    lines = (llm.LOGS_DIR / "usage.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["model"] == "test/chat"
    assert record["node"] == "triage"
    assert (record["prompt_tokens"], record["completion_tokens"], record["total_tokens"]) == (
        10,
        5,
        15,
    )
    assert record["latency_ms"] >= 0
    assert llm.summarize_usage() == {
        "calls": 1,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "by_model": {"test/chat": {"calls": 1, "prompt_tokens": 10, "completion_tokens": 5}},
    }


def test_ocr_cache_new_key_then_legacy_then_offline(monkeypatch):
    image = llm.CACHE_DIR / "scan.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"not really a png")
    keyed, legacy = llm.ocr_cache_paths(image.read_bytes())
    assert keyed != legacy and keyed.parent == legacy.parent

    with pytest.raises(RuntimeError):
        llm.transcribe_image(image, offline=True)          # nothing cached, no model call allowed

    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("legacy transcription", encoding="utf-8")
    assert llm.transcribe_image(image, offline=True) == "legacy transcription"

    keyed.write_text("keyed transcription", encoding="utf-8")
    assert llm.transcribe_image(image, offline=True) == "keyed transcription"   # new key wins


def test_ocr_cache_key_changes_with_vision_model(monkeypatch):
    keyed_a, _ = llm.ocr_cache_paths(b"img")
    monkeypatch.setattr(llm.get_settings(), "vision_model", "other/vision")
    keyed_b, _ = llm.ocr_cache_paths(b"img")
    assert keyed_a != keyed_b


def test_usage_logger_falls_back_to_message_usage_metadata():
    logger = llm.UsageLogger()
    run_id = uuid4()
    message = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
    )
    result = LLMResult(generations=[[ChatGeneration(message=message)]])

    logger.on_llm_end(result, run_id=run_id)

    record = json.loads((llm.LOGS_DIR / "usage.jsonl").read_text(encoding="utf-8"))
    assert (record["prompt_tokens"], record["completion_tokens"]) == (7, 3)
    assert record["node"] is None
