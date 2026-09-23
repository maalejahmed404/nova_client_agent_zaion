import hashlib
import json
import time
from base64 import b64encode
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import openai
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_random_exponential

from .config import CACHE_DIR, CORPUS_DIR, LOGS_DIR, get_settings, now

RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})

OCR_PROMPT = (
    "Transcris fidèlement ce document en markdown. "
    "Les tableaux restent des tableaux markdown, "
    "l'en-tête du document (référence, date de mise à jour, mention de remplacement) est conservé, "
    "pas de commentaire, pas de résumé. "
    "Donne uniquement la transcription."
)


class HTTPError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(message)


def _should_retry(exc: BaseException) -> bool:
    # APITimeoutError hérite de APIConnectionError ; RateLimitError et
    # InternalServerError (5xx) héritent de APIStatusError.
    if isinstance(exc, (openai.APIConnectionError, httpx.TransportError)):
        return True
    if isinstance(exc, (openai.APIStatusError, HTTPError)):
        return exc.status_code in RETRYABLE_STATUS
    return False


def retry_policy() -> Retrying:
    """Backoff exponentiel avec jitter entre 1 et 30 s, 5 tentatives."""
    return Retrying(
        retry=retry_if_exception(_should_retry),
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )


def invoke_with_retry(runnable: Runnable, input: Any) -> Any:
    return retry_policy()(runnable.invoke, input)


def create_chat_client(
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> ChatOpenAI:
    """Client OpenRouter déterministe. Le fallback se fait côté serveur ; le client ne réessaie jamais."""
    settings = get_settings()
    model = model or settings.chat_model
    return ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens if max_tokens is not None else settings.max_tokens,
        max_retries=0,
        extra_body={"models": [model, settings.chat_model_fallback]},
        **kwargs,
    )


def structured_answer(
    messages: list[BaseMessage],
    output_model: type[BaseModel],
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> BaseModel:
    """Extraction par function calling : json_schema n'est pas supporté par tous les modèles OpenRouter."""
    llm = create_chat_client(model, temperature, max_tokens).with_structured_output(
        output_model, method="function_calling"
    )
    result = invoke_with_retry(llm, messages)
    if result is None:
        raise ValueError(f"Le modèle n'a pas appelé l'outil {output_model.__name__}.")
    return result


@lru_cache(maxsize=1)
def _http() -> httpx.Client:
    settings = get_settings()
    return httpx.Client(
        base_url=settings.openrouter_base_url,
        headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
    )


def _post(path: str, payload: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
    response = _http().post(path, json=payload, timeout=timeout)
    if response.status_code >= 400:
        raise HTTPError(response.status_code, f"{path} : {response.text[:500]}")
    return response.json()


def embed_texts(
    texts: list[str],
    model: str | None = None,
    query_instruction: str | None = None,
    batch_size: int = 100,
) -> list[list[float]]:
    """Vecteurs dans l'ordre des textes. Cache disque, un fichier par texte et par modèle."""
    model = model or get_settings().embedding_model
    cache_dir = CACHE_DIR / "embeddings" / model
    cache_dir.mkdir(parents=True, exist_ok=True)

    files = {
        text: cache_dir
        / f"{hashlib.sha256(f'{model}:{query_instruction or ""}:{text}'.encode()).hexdigest()}.json"
        for text in dict.fromkeys(texts)
    }
    vectors = {
        t: json.loads(f.read_text(encoding="utf-8"))["embedding"]
        for t, f in files.items()
        if f.exists()
    }
    missing = [t for t in files if t not in vectors]

    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        inputs = (
            [f"Instruct: {query_instruction}\nQuery: {t}" for t in batch]
            if query_instruction
            else batch
        )
        data = retry_policy()(_post, "embeddings", {"model": model, "input": inputs})
        items = sorted(data["data"], key=lambda item: item["index"])
        if len(items) != len(batch):
            raise ValueError(f"{len(batch)} textes envoyés, {len(items)} vecteurs reçus.")
        for text, item in zip(batch, items):
            vectors[text] = item["embedding"]
            tmp = files[text].with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"text": text, "embedding": item["embedding"]}), encoding="utf-8"
            )
            tmp.replace(files[text])

    return [vectors[t] for t in texts]


def image_to_markdown(
    image_path: str | Path, model: str | None = None, offline: bool = False
) -> str:
    """Transcription markdown fidèle. Cache indexé sur le contenu de l'image, le modèle et le prompt."""
    path = Path(image_path)
    if not path.is_absolute():
        path = CORPUS_DIR / path
    image = path.read_bytes()
    model = model or get_settings().vision_model

    key = hashlib.sha256(image + f"\0{model}\0{OCR_PROMPT}".encode()).hexdigest()
    cache_file = CACHE_DIR / "ocr" / f"{key}.md"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    if offline:
        raise RuntimeError(f"Aucune transcription en cache pour {path} avec {model}.")

    mime = {".png": "image/png", ".webp": "image/webp"}.get(path.suffix.lower(), "image/jpeg")
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": get_settings().max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64encode(image).decode()}"},
                    },
                ],
            }
        ],
    }
    choice = retry_policy()(_post, "chat/completions", payload, 180.0)["choices"][0]
    if choice.get("finish_reason") == "length":
        raise RuntimeError(f"Transcription tronquée pour {path} : augmenter max_tokens.")

    text = choice["message"]["content"]
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(cache_file)
    return text


class CostLoggerCallback(BaseCallbackHandler):
    """Une ligne JSONL par appel de modèle : modèle, nœud, tokens, latence."""

    def __init__(self, graph_node: str = "unknown") -> None:
        super().__init__()
        self.graph_node = graph_node
        self.starts: dict[UUID, float] = {}
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self.starts[run_id] = time.perf_counter()

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self.starts.pop(run_id, None)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        latency_ms = int((time.perf_counter() - self.starts.pop(run_id)) * 1000)
        message = response.generations[0][0].message
        usage = message.usage_metadata or {}
        entry = {
            "timestamp": now().isoformat(),
            "model": message.response_metadata.get("model_name", "unknown"),
            "node": self.graph_node,
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "latency_ms": latency_ms,
        }
        with open(LOGS_DIR / "llm_costs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def read_cost_summary() -> dict[str, Any]:
    """Totaux globaux et par modèle ; zéros si rien n'est journalisé."""
    fields = ("calls", "prompt_tokens", "completion_tokens", "total_tokens", "latency_ms")
    overall = dict.fromkeys(fields, 0)
    by_model: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(fields, 0))

    log_file = LOGS_DIR / "llm_costs.jsonl"
    lines = log_file.read_text(encoding="utf-8").splitlines() if log_file.exists() else []
    for line in filter(str.strip, lines):
        entry = json.loads(line) | {"calls": 1}
        for stats in (overall, by_model[entry["model"]]):
            for k in fields:
                stats[k] += entry[k]

    return {"overall": {f"total_{k}": v for k, v in overall.items()}, "by_model": dict(by_model)}
