# Project rules for the coding assistant

Néova Télécom customer-relations agent. Python 3.12 managed with uv. Package in src/neova.
Layout: src/neova/config.py (settings + clock + paths), src/neova/api (FastAPI),
src/neova/llm.py (OpenRouter gateway), src/neova/rag (retrieval),
src/neova/agent (LangGraph; agent/tools.py = HTTP client of the API).
Data: data/neova_data.json. Corpus: corpus/*.pdf and one .png.
This machine is Windows: PowerShell syntax, always through `uv run`.

## Scope of each task
- Read only the files named in the prompt. Never open other files, never browse git history,
  never read deleted files.
- Change only the files named in the prompt.
- Never run the test suite or any long command; I run them and paste what fails.
- Answer with the diff. No summary, no explanation of what you just wrote.

## Hard rules
- Never call datetime.now(). Always `from neova.config import now` — the dataset is frozen on
  2026-08-25 and REFERENCE_NOW freezes the clock there.
- Never hardcode a model id, an API key or a URL. Read neova.config.get_settings().
- Build every path from PROJECT_ROOT / DATA_DIR / CORPUS_DIR / CACHE_DIR / LOGS_DIR.
- Never read, print or modify .env. Never run git commit.
- Every LLM or embedding call goes through src/neova/llm.py.
- Writes to the API always carry an Idempotency-Key.
- User-facing strings are French; code, identifiers and comments are English.

## Style
- Type hints on every signature. Pydantic models at the API boundary, dataclasses inside.
- No comment that restates the code. A comment earns its place only when it records a
  decision or a non-obvious constraint ("json_schema is unsupported by some models").
- No abstraction used once: no helper, wrapper, base class or config flag with a single
  caller. Inline it.
- No silent `except Exception`. Catch the exception you expect, or let it rise. An error
  message names what failed and what the caller can do.
- No defensive `if x is None` on values that cannot be None, no fallback that hides a bug.
- Guard clauses over nested ifs. A function fits on a screen.
- Prefer the standard library. Add no dependency without being asked.
- Tests assert behaviour through the public function, never internals. One test, one claim.
  No mock of the thing under test.
- Line length 100 (ruff), target py312.