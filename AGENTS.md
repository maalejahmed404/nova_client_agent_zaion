# Project rules for the coding assistant

Néova Télécom customer-relations agent. Python 3.12 managed with uv. Package in src/neova.
Layout: src/neova/config.py (settings + clock + paths),
src/neova/api (FastAPI), src/neova/llm.py (OpenRouter gateway), src/neova/rag (retrieval),
src/neova/agent (LangGraph; agent/tools.py = HTTP client of the API). Data: data/neova_data.json. Corpus: corpus/*.pdf and one .png.
This machine is Windows: run commands with PowerShell syntax, always through `uv run`.

Hard rules:
- Never call datetime.now(). Always `from neova.config import now` — the dataset is frozen on
  2026-08-25 and REFERENCE_NOW in .env freezes the clock there.
- Never hardcode a model id, an API key or a URL. Read neova.config.get_settings().
- Build every path from neova.config.PROJECT_ROOT / DATA_DIR / CORPUS_DIR / CACHE_DIR /
  LOGS_DIR, never from the current working directory.
- Never read, print or modify .env. Never run git commit.
- Every LLM or embedding call goes through src/neova/llm.py once it exists.
- User-facing strings are French; code, identifiers and comments are English.
- No comments that explain what the code does. No abstraction used only once. No silent
  `except Exception`.
- After writing code run `uv run pytest -q` and paste the result. Keep answers short; do not
  summarise files back to me.
