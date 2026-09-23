# Project rules for the coding assistant

Néova Télécom customer-relations agent. Python 3.12 managed with uv. Package in src/neova.
Layout: src/neova/config.py (settings + clock + paths), src/neova/api (FastAPI),
src/neova/llm.py (OpenRouter gateway), src/neova/rag (retrieval),
src/neova/agent (LangGraph; agent/client.py = HTTP client of the API).
Data: data/neova_data.json. Corpus: corpus/*.pdf and one .png.
This machine is Windows: PowerShell syntax, always through `uv run`.

## Scope of each task
- Read only the files named in the prompt. Never open other files, never browse git history,
  never read deleted files.
- Change only the files named in the prompt.
- Never run the test suite or any long command; I run them and paste what fails.
- Answer with the diff. No summary, no explanation of what you just wrote.
- site the files you changed when you finish.
## Hard rules
- Never call datetime.now(). Always `from neova.config import now` — the dataset is frozen on
  2026-08-25 and REFERENCE_NOW freezes the clock there.
- Never hardcode a model id, an API key or a URL. Read neova.config.get_settings().
- Build every path from PROJECT_ROOT / DATA_DIR / CORPUS_DIR / CACHE_DIR / LOGS_DIR.
- Never read, print or modify .env. Never run git commit.
- Every LLM or embedding call goes through src/neova/llm.py.
- Writes to the API always carry an Idempotency-Key.
- User-facing strings are French; code, identifiers and comments are English.

## How the agent graph works
- Six nodes, named in French: precheck, agent, outils, postreview, escalade, fin.
  precheck goes to agent or escalade. agent goes to outils or postreview. outils goes to agent
  or postreview depending on after_tools. postreview goes to outils, agent, escalade or fin. escalade goes to fin.
- Every node sets state["next"] explicitly before returning. A router that finds no choice
  raises, it never falls back on the previous value.
- Any node that routes to outils sets after_tools at the same time, naming the node outils
  must return to.
- When outils runs for postreview, it executes only chercher_documentation for a document
  need and dossier_client for a fact need, never a write tool, and it appends a synthetic
  assistant tool call before the tool message so the history stays valid for the provider.
- precheck is a structured call over precheck.txt: it decides only whether the request must
  be transferred before any treatment.
- agent is the ReAct loop: the chat model with the seven tools bound. Tool bodies are empty;
  the outils node executes them, because only it sees the state.
- The model picks the reads, the code owns the writes. confirmer_rendez_vous is never
  executed until a separate structured call over confirmation.txt says the customer clearly
  accepted; otherwise the tool answers that no confirmation was given.
- A booking carries the idempotency key stored with the proposal. If the API fails, the key
  stays in the state and is replayed next turn, so a retry can never book twice.
- Eight tool calls per turn at most; beyond that the turn goes to postreview.
- postreview is a structured call over post_review.txt. It may ask for what it misses, and
  the code fetches it, but only one such round per turn. If that call itself fails, go to
  escalade: transferring beats answering wrong.
- escalade builds the ticket with ticket.txt, posts it, then writes the message with
  handoff_message.txt, filling <delai> from the API's callback_eta. If the API is down,
  announce the transfer without a delay rather than inventing one. Never escalate twice in
  the same turn.
- The agent never announces its own transfer: it does not decide it.
- When the customer verifies a different identity, wipe session, customer_id, facts,
  proposal and passages. Keep the messages.
- After two failed identity checks, stop insisting and escalate.

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
- All imports at the top of the file, never inside a function.
- Tests patch a constant in the module that uses it, not only in neova.config:
  `from neova.config import CACHE_DIR` copies the value at import time.