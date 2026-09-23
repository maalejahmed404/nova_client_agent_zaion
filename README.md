# Néova, customer-relations agent

A LangGraph agent that answers Néova Télécom customers in French: retrieval over `corpus/`, a
FastAPI service over `data/` (with a state-changing booking endpoint), and a transfer path to a
human advisor.

## Run it

```bash
uv run neova chat
```

## Models

Chat `google/gemini-2.5-flash`, fallback `mistralai/mistral-small-3.2-24b-instruct`, vision
`google/gemini-2.5-flash` for the scanned roaming sheet, all set in environment variables. The
clock is frozen on the dataset's date (`REFERENCE_NOW`, 25/08/2026 10:00).

**Embeddings: `qwen/qwen3-embedding-8b`.** Chosen for French: multilingual by training (first on
MTEB multilingual at release), and a query-side instruction bridges colloquial customer French
and the formal wording of the documents; paired with French-stemmed BM25 for exact terms.
Measured full-evidence recall on the 45-question French set is 0.95; a side-by-side comparison
with another model was not run.

## The graph

![graph](docs/graph.png)

Dotted arrows are conditional: each node names the next one. `postreview` goes to `outils` to fetch
a missing piece: a document returns to `agent`, which rewrites; a customer fact returns to
`postreview`.

- **precheck**: if the customer's message holds a customer number and a phone, code verifies them
  with the API and stores the customer file in the state. An LLM then checks the seven "transfert
  immédiat" situations of the escalation procedure (GDPR, legal threat, death, fraud, Pro contract,
  minor, distress) on the new message and the file. Its quote must appear in the customer's
  messages or in the file, otherwise it is ignored. A situation already transferred in the
  conversation gets a fixed reminder and no new ticket. If this LLM call fails, the message goes
  to the agent.
- **agent and outils**: ReAct. The chat model has 7 tools bound (`chercher_documentation`,
  `verifier_identite`, `dossier_client`, `incidents_zone`, `proposer_rendez_vous`,
  `confirmer_rendez_vous`, `verifier_geste_commercial`). Tool bodies are empty: the model only
  requests them, and `outils` executes each request with its own checks. After 8 tool calls in a
  turn, further requests are dropped and the turn goes to postreview.
- **Booking**: a proposal needs one of the four documented cases, a fibre plan, and, for the cases
  "persistent red light" and "repeated cuts", no incident already started in the zone. A booking
  needs a customer message sent after the proposal, then a separate LLM check that the customer
  accepted, which reads the agent's last message and rejects the acceptance if the 69 € condition
  was not announced there. If the booking API fails, the proposal and its idempotency key stay in
  the state; the next confirmation reuses the same key without asking the customer again, so a
  retry never books twice.
- **Commercial gesture**: a separate LLM applies the internal policy to the customer's request and
  file. The agent only receives the outcome to tell the customer. When the policy sends the case
  to level 2, puts it out of scope, or cannot decide, the graph goes straight to `escalade`.
- **postreview**: every claim of the agent's reply needs a source: customer facts in the file or in
  this turn's tool results, rules in the retrieved documents. It may fetch one missing piece per
  turn. If the reply is not covered, no documentation search ran this turn and that fetch is still
  unused, code runs a search and the agent rewrites. A reply still not covered, a "transfert après
  examen" case, or a failure of this LLM call goes to `escalade`.
- **escalade**: an LLM drafts the ticket, code posts it, and another LLM writes the transfer
  message with the callback delay returned by the API. If the ticket API is down, the transfer is
  announced without a delay rather than an invented one.

State lives in `MemorySaver`, one thread per conversation; it does not survive a restart.

## Evaluation

**Retrieval.** 45 hand-written French questions (37 answerable, 8 not),
`uv run python -m neova.rag.eval_retrieval`:

| Measure | Result |
|---|---|
| Full-evidence recall (every needed section retrieved) | **0.95** |
| Any-evidence recall | 0.97 |
| MRR of the first expected section | 0.82 |
| Archived sections returned without a reason | **0** |

The two misses: a "lots of data" mobile question buried under the roaming sheet, and a gold line
whose contract legitimately reaches the archived promo. Without the status filter, "combien coûte
la fibre 1 Gb/s" would return the 2024 promo price (24,99 €): it wins both rankings. A similarity
threshold cannot detect unanswerable questions: the two score ranges overlap.

**Agent.** The agent was evaluated through manual conversations rather than an automated
pipeline. Retrieval has a single correct answer per question and lends itself to a metric; the
agent's quality lies in whole exchanges (which tool it calls, when it transfers, what it tells the
customer), which I found more reliable to judge by reading them. Testing concentrated on the flows
that carry risk: network failures, technician booking, billing, immediate and reviewed transfers,
and contradictory documents. The representative scenarios below all pass; run each in a new chat,
typing a customer's full name to identify:

| Customer | Message(s) | Expected |
|---|---|---|
| `Camille Rousseau` | `Je n'ai plus internet, le voyant clignote en rouge.` | Outage in her area, estimated back at 18:00; no reboot, no technician |
| `Patrick Doré` | `Mon voyant est rouge fixe.` | No incident in Lille: the restart procedure |
| `Léa Nguyen` | `Mon voyant est orange.` | Degraded connection; may mention tomorrow's planned maintenance |
| | `Je veux une copie de toutes les données que vous avez sur moi.` | Immediate transfer, no question asked |
| | `Supprimez mon compte et toutes mes données, j'invoque le RGPD.` | Immediate transfer |
| | `Je vais saisir le médiateur des communications électroniques.` | Immediate transfer |
| `Patrick Doré` | `Mon voyant est rouge fixe, j'ai déjà redémarré deux fois à dix minutes d'intervalle.` → `oui` | Friday 28/08 9h-11h proposed with the 69 € condition; booked only after `oui` |
| `Patrick Doré` | same first message → `Pas ce créneau.` | Monday 31/08 14h-16h proposed |
| | `La fibre 1 Gb/s est toujours en promo à 24,99 € ?` | No: the 2024 offer is over, current price 39,99 € |
| | `Quels étaient les prix de la promo de rentrée 2024 ?` | 2024 prices, presented as a past offer |
| | `Combien coûte la fibre 1 Gb/s ?` | 39,99 €/month, 12 or 24-month commitment |
| | `Combien coûte l'option décodeur TV ?` | 5 €/month (a table whose columns are shifted in the PDF) |

**Limits of this evaluation.** Coverage is deliberate, not exhaustive: less frequent paths
(identity changes mid-conversation, API outages during a booking, rarer transfer situations) were
checked only occasionally, and a systematic pass over every scenario was out of reach in the time
available. Without a pass rate, the results are qualitative. The failures that remain are few and
share two roots: prompt instructions the model does not always follow, and the model's own
interpretation. In practice it can explain an amount by listing possible causes instead of
computing it from the customer file, repeat a question already answered, or answer before
searching the documentation. Each of these is caught by `postreview` only when it leaves a claim
without a source.

## Three design decisions

**1. ReAct with code-owned writes, not a state machine.** I first designed a state machine: safer,
and the LLM never decides when a tool is needed. Covering every case that way proved too complex in
the time available, so the model chooses its reads and code guards every write (proposal
conditions, explicit acceptance, idempotency, replay). *Traded away:* a deterministic path: the
model sometimes skips a tool it should call, as noted in the evaluation.

**2. When to escalate: two LLM gates around the agent, and no answer without a source.** The
escalation procedure splits into cases to transfer before any work and cases to handle first, then
transfer if the rules do not settle them. The graph mirrors that split: `precheck` applies the
first list before the agent runs, `postreview` the second one after it has written its reply. The
same gate settles unanswerable questions: every claim in a reply must come from the customer file,
a tool result or a retrieved document, and a reply that cannot be sourced is transferred rather
than sent. The procedure and the gesture policy are internal documents, never indexed, so the
agent cannot quote their thresholds; they are copied into the prompts of the nodes that apply
them, and the conditions of a technician visit are written in code, which was reliable given how
small these documents are. *Traded away:* cost, latency and some over-transfer. Each message costs
three to six LLM calls, and a correct reply the gate cannot source is transferred anyway. A
changed document also means editing prompts or code.

**3. Contradictions resolved by document status before search.** Each chunk carries its
document's status, dates and offer window; deprecated chunks are filtered out unless the question
is historical or the customer's contract started inside the offer window. *Traded away:* a future
obsolete document without a status header would not be filtered.

## What's broken

- **LLM misbehaviour grows with the conversation.** As the history lengthens, the model invents
  calculations, repeats questions already answered, offers actions its tools cannot do, and does
  not always chain tools. Prompts are the main lever and must be very well written; they are not
  tuned enough yet.
- postreview checks that claims have a source, not that they are correct, and lets any question
  to the customer through.
- The rewrite path leaves the rejected draft in the history.
- Not production-grade: one HTTP client and one API session shared by every conversation (safe
  with one chat per process, not with concurrent ones), in-memory state only, and the design leans
  on the state to remember what was done rather than on persisted records.
- precheck compares the quote character by character: a curly apostrophe can cancel a mandatory
  transfer. Identifiers written `+33 …` are not recognised by precheck.
- Transfers decided by postreview are not remembered: repeating the request opens a second ticket.
- The FAQ says technicians work Tuesday to Saturday; the API offers a Monday slot and is treated as
  the system of record. There is no cancellation tool.

## With two more days

1. A design that covers every known flow explicitly (failure, booking, billing, termination,
   moving) as states, keeping the free agent for open questions.
2. Test-driven prompt work: run the scenarios automatically, fix the prompt behind each failure,
   measure again. Tests are what expose a weak prompt.
3. An automated end-to-end evaluation with a pass rate and error analysis, traced in Langfuse.
4. Production practices as a priority: one API session per conversation, a persistent
   checkpointer, session expiry, persisted records of actions.
5. A correctness check in postreview, and a rewrite path that drops the rejected draft.

Spend: about $9.7 of the $10, coding assistant included.
