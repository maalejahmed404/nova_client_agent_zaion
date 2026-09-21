from datetime import datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic_settings import BaseSettings, SettingsConfigDict

PARIS = ZoneInfo("Europe/Paris")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_DIR = PROJECT_ROOT / "corpus"
CACHE_DIR = PROJECT_ROOT / ".cache"
LOGS_DIR = PROJECT_ROOT / "logs"

# Assumption, not in the escalation procedure: it gives the delays, not the opening hours.
BUSINESS_HOURS = (9, 18)
BUSINESS_DAYS = range(0, 5)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    chat_model: str
    chat_model_fallback: str
    judge_model: str
    embedding_model: str
    vision_model: str
    max_tokens: int = 800

    reference_now: str | None = None

    api_base_url: str = "http://localhost:8000"
    api_chaos_rate: float = 0.0

    grounding_check: bool = False
    refusal_threshold: float = 0.30

    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def now() -> datetime:
    """The only clock in the project. REFERENCE_NOW freezes it on the dataset's date."""
    ref = get_settings().reference_now
    if ref:
        return datetime.fromisoformat(ref).astimezone(PARIS)
    return datetime.now(PARIS)


def clock_mode() -> str:
    return "simulated" if get_settings().reference_now else "wall"
