import base64
import hashlib
import json
import time
from pathlib import Path
from uuid import UUID

import httpx
import openai
import tenacity
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from neova.config import CACHE_DIR, LOGS_DIR, get_settings

RETRYABLE_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
EMBEDDING_BATCH_SIZE = 32
OCR_PROMPT = (
    "Transcris fidèlement ce document en markdown. Conserve les tableaux en tableaux markdown. "
    "Conserve l'en-tête (référence, date de mise à jour, mention de remplacement). "
    "Ne commente pas, ne résume pas."
)


class RetryableHTTPStatus(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"retryable HTTP status {status_code}")
        self.status_code = status_code


RETRY = tenacity.retry(
    retry=tenacity.retry_if_exception_type(
        (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
            httpx.TransportError,
            RetryableHTTPStatus,
        )
    ),
    wait=tenacity.wait_random_exponential(min=1, max=30),
    stop=tenacity.stop_after_attempt(5),
    reraise=True,
)


def get_chat(model: str | None = None, **kwargs) -> ChatOpenAI:
    settings = get_settings()
    primary = model or settings.chat_model
    params = {
        "base_url": settings.openrouter_base_url,
        "api_key": settings.openrouter_api_key,
        "model": primary,
        "temperature": 0,
        "max_tokens": settings.max_tokens,
        "max_retries": 0,
        "default_headers": {"X-Title": "neova-agent"},
        "extra_body": {"models": [primary, settings.chat_model_fallback]},
    }
    params.update(kwargs)
    return ChatOpenAI(**params)


@RETRY
def invoke_with_retry(runnable, input, **kwargs):
    return runnable.invoke(input, **kwargs)


@RETRY
def structured_call[T: BaseModel](messages, schema: type[T], model: str | None = None) -> T:
    # function_calling: not every OpenRouter model supports json_schema response formats
    structured = get_chat(model).with_structured_output(schema, method="function_calling")
    return structured.invoke(messages)


@RETRY
def _post_embeddings(payload: dict) -> list[list[float]]:
    settings = get_settings()
    response = httpx.post(
        f"{settings.openrouter_base_url.rstrip('/')}/embeddings",
        json=payload,
        headers={
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "X-Title": "neova-agent",
        },
        timeout=60,
    )
    if response.status_code in RETRYABLE_STATUSES:
        raise RetryableHTTPStatus(response.status_code)
    response.raise_for_status()
    items = sorted(response.json()["data"], key=lambda item: item.get("index", 0))
    return [item["embedding"] for item in items]


def embed_texts(texts: list[str], instruction: str | None = None) -> list[list[float]]:
    settings = get_settings()
    model_dir = settings.embedding_model.replace("/", "_").replace(":", "_")
    cache_dir = CACHE_DIR / "embeddings" / model_dir
    sent = [f"Instruct: {instruction}\nQuery: {text}" if instruction else text for text in texts]
    paths = [cache_dir / f"{hashlib.sha256(s.encode('utf-8')).hexdigest()}.json" for s in sent]

    vectors: dict[int, list[float]] = {}
    missing: list[int] = []
    for index, path in enumerate(paths):
        if path.exists():
            vectors[index] = json.loads(path.read_text(encoding="utf-8"))
        else:
            missing.append(index)

    for start in range(0, len(missing), EMBEDDING_BATCH_SIZE):
        batch = missing[start : start + EMBEDDING_BATCH_SIZE]
        batch_vectors = _post_embeddings(
            {"model": settings.embedding_model, "input": [sent[i] for i in batch]}
        )
        for index, vector in zip(batch, batch_vectors, strict=True):
            vectors[index] = vector
            paths[index].parent.mkdir(parents=True, exist_ok=True)
            paths[index].write_text(json.dumps(vector), encoding="utf-8")

    return [vectors[index] for index in range(len(sent))]


OCR_PROMPT_VERSION = "1"


def ocr_cache_paths(image_bytes: bytes) -> tuple[Path, Path]:
    """(new key: image + vision model + prompt version, legacy key: image only)."""
    settings = get_settings()
    keyed = hashlib.sha256(
        image_bytes + settings.vision_model.encode("utf-8") + OCR_PROMPT_VERSION.encode("utf-8")
    ).hexdigest()
    legacy = hashlib.sha256(image_bytes).hexdigest()
    return CACHE_DIR / "ocr" / f"{keyed}.md", CACHE_DIR / "ocr" / f"{legacy}.md"


def transcribe_image(path: Path, offline: bool = False) -> str:
    image_bytes = path.read_bytes()
    cache_file, legacy_file = ocr_cache_paths(image_bytes)
    for candidate in (cache_file, legacy_file):
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    if offline:
        raise RuntimeError(f"no cached transcription for {path.name} and offline mode is on")

    encoded = base64.b64encode(image_bytes).decode("ascii")
    message = HumanMessage(
        content=[
            {"type": "text", "text": OCR_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
        ]
    )
    response = invoke_with_retry(get_chat(get_settings().vision_model), [message])
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(response.content, encoding="utf-8")
    return response.content


def _usage_from_messages(response: LLMResult) -> dict[str, int]:
    for generations in response.generations:
        for generation in generations:
            usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
            if usage:
                prompt = usage.get("input_tokens", 0)
                completion = usage.get("output_tokens", 0)
                return {
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": usage.get("total_tokens", prompt + completion),
                }
    return {}


class UsageLogger(BaseCallbackHandler):
    def __init__(self) -> None:
        self._starts: dict[UUID, tuple[float, str | None]] = {}

    def on_llm_start(self, serialized, prompts, *, run_id: UUID, **kwargs) -> None:
        params = kwargs.get("invocation_params") or {}
        model = params.get("model") or params.get("model_name")
        self._starts[run_id] = (time.perf_counter(), model)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, tags=None, **kwargs) -> None:
        started, start_model = self._starts.pop(run_id, (time.perf_counter(), None))
        llm_output = response.llm_output or {}
        usage = llm_output.get("token_usage") or _usage_from_messages(response)
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        record = {
            "ts": time.time(),
            "model": llm_output.get("model_name") or start_model,
            "node": tags[0] if tags else None,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": usage.get("total_tokens", prompt + completion),
            "latency_ms": round((time.perf_counter() - started) * 1000),
        }
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with (LOGS_DIR / "usage.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(record) + "\n")


def summarize_usage() -> dict:
    summary = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "by_model": {}}
    log_path = LOGS_DIR / "usage.jsonl"
    if not log_path.exists():
        return summary
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        per_model = summary["by_model"].setdefault(
            record.get("model") or "unknown",
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
        )
        for bucket in (summary, per_model):
            bucket["calls"] += 1
            bucket["prompt_tokens"] += record["prompt_tokens"]
            bucket["completion_tokens"] += record["completion_tokens"]
    return summary
