# InsightPDF — Agentic RAG Document Intelligence Platform

InsightPDF is a document intelligence platform, not a PDF chatbot. Upload one or many
PDFs into a workspace and it classifies each one, builds a structured profile, exposes
the tools that actually make sense for that document (a research paper gets dataset/
methodology/results analysis; a novel gets spoiler-controlled character and theme
analysis; a textbook gets quizzes and question-paper generation), and answers follow-up
questions through an agentic retrieval loop that cites its sources down to the page.

**InsightPDF is designed to work without paid LLM API usage**, using configured free
hosted models and automatic provider failover. RAG is one capability of the system, not
the whole product.

```
                    INSIGHTPDF
                         │
                  Document Workspace
                         │
             ┌───────────┴───────────┐
             │                       │
        Document Intelligence    Multi-Document
             │                    Analysis
             ↓                       ↓
       Classification            Compare
       Capabilities              Research
       Extraction                Synthesis
             │                       │
             └───────────┬───────────┘
                         ↓
                  Agentic Workflows
                         │
            ┌────────────┼────────────┐
            ↓            ↓            ↓
          RAG         Tools        Web
            │            │            │
            └────────────┼────────────┘
                         ↓
                    Validation
                         ↓
               Evidence + Citations
                         ↓
                    User Output
```

## Contents

- [Architecture](#architecture)
- [Supported document types & capabilities](#supported-document-types--capabilities)
- [Agentic RAG](#agentic-rag)
- [Tools](#tools)
- [Free LLM gateway & automatic failover](#free-llm-gateway--automatic-failover)
- [Vector store abstraction](#vector-store-abstraction)
- [Evaluation & observability](#evaluation--observability)
- [Dark mode](#dark-mode)
- [Setup](#setup)
- [Environment variables](#environment-variables)
- [Example workflows](#example-workflows)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Known limitations & honest scoping notes](#known-limitations--honest-scoping-notes)

## Architecture

```
Next.js (TypeScript)
   ↓  fetch / SSE
FastAPI
   ↓
Services (workspace, document, tool, chat)
   ↓
Agents (LangGraph) / Workflows / Tools
   ↓
PostgreSQL + Vector Store (Chroma/Qdrant) + File Storage
   ↓
LLM Gateway (Groq ⇄ Gemini ⇄ OpenRouter, free-first) + Embeddings (local HuggingFace)
```

The frontend never talks to an LLM or a database directly — every request goes through
the FastAPI layer. No API keys are ever sent to the browser (`GET /api/config/public`
only exposes non-secret, derived flags).

### Backend layout

```
backend/app/
├── api/routes/       FastAPI routers (workspaces, documents, tools, chat, evaluation, models, config)
├── agents/           LangGraph state graph + the DocumentAgent wrapper
├── tools/             BaseTool abstraction, the generic PromptTool executor, and per-mode catalogs
├── ingestion/          PDF parsing/validation, structure detection, chunking, the processing pipeline
├── classification/     DocumentProfile schema + the type→capability map + the LLM classifier
├── extraction/         Table extraction (pdfplumber)
├── retrieval/           Retriever, query rewriter, evidence/citation validator
├── vectorstores/       VectorStore interface + Chroma/Qdrant implementations
├── embeddings/          EmbeddingProvider interface + local HuggingFace implementation
├── llm/                 LLMProvider interface, the free-provider-first ModelGateway, task→tier router
├── evaluation/          Real metric computation + the run tracker
├── services/            Business logic between the API and the above layers
├── models/               SQLAlchemy ORM (Postgres)
├── schemas/              Pydantic request/response + shared types (Citation, ToolResultEnvelope)
└── core/                  config, logging, exceptions, enums
```

### Frontend layout

```
frontend/app/
├── page.tsx                                        Dashboard — workspace list/create
├── models/page.tsx                                 Free-provider status (Primary/Secondary/Tertiary)
├── evaluation/page.tsx                              Evaluation dashboard
├── settings/page.tsx                                Read-only config
└── workspaces/[workspaceId]/
    ├── page.tsx                                     Upload + document library (independent per-file state)
    ├── documents/[documentId]/page.tsx               Profile + dynamic tool panel + results + follow-up chat
    ├── chat/page.tsx                                 Workspace-wide chat (document scope + spoiler level)
    ├── compare/page.tsx                              Multi-document selection + comparison tools
    └── research/page.tsx                             Web research tool
frontend/components/                                  ToolPanel, ToolResultView, ChatThread (SSE streaming),
                                                        Citations, DocumentCard (Retry/Delete), ThemeToggle, …
```

## Supported document types & capabilities

On upload, the classifier produces a structured `DocumentProfile`:

```json
{
  "type": "research_paper",
  "confidence": 0.94,
  "title": "...",
  "sections": [{ "title": "Abstract", "page_start": 1, "page_end": 1 }],
  "entities": [{ "name": "WMT 2014", "type": "dataset", "mentions": 3 }],
  "topics": ["attention mechanisms", "sequence modeling"],
  "capabilities": ["summarize", "chat", "analyze_dataset", "compare_papers", "..."]
}
```

`capabilities` is a plain list of tool ids, computed once from
`app/classification/profiles.py::CAPABILITIES_BY_TYPE`. The frontend's tool panel
renders whichever tools from the full catalog (`GET /api/tools/catalog`) appear in a
document's own `capabilities` — there is no `if document_type == "research_paper"`
branch anywhere in the UI.

Six document types are supported: `research_paper`, `literature`, `study_material`,
`technical_documentation`, `business_report`, and `generic`.

## Agentic RAG

Chat doesn't do `question → retrieve → answer`. It's a LangGraph state machine:

```
User message
     │
Understand intent  ───────────────► known tool requested? ──► run that tool
     │ no                                                          │
     ▼                                                             │
Rewrite / expand query                                             │
     │                                                             │
     ▼                                                             │
Retrieve evidence ◄────────────────┐                                │
     │                              │                                │
Enough evidence? (similarity        │                                │
score + coverage heuristic)         │                                │
     │ no                           │                                │
     ▼                              │                                │
Refine query ───────────────────────┘ (bounded by RETRIEVAL_MAX_REFINE_ATTEMPTS)
     │ yes
     ▼
Analyze (grounded, cited answer)
     │
Validate citations (excerpt-vs-source check)
     │
     ▼
Answer + citations ◄─────────────────────────────────────────────────┘
```

Retrieval sufficiency is judged by an actual similarity-score heuristic, not another LLM
call. Tool selection is also **not** done via each provider's native function-calling API
— it's a constrained structured-output classification validated against the document's
actual registered capability list (`decision.tool_id in available`). This is a
deliberate choice: free-tier models/providers have inconsistent native tool-calling
support, while this approach works identically regardless of which of the three
providers answered the request, and it's impossible for the model to invoke a tool that
isn't registered and available for that document.

## Tools

Two layers, one registry (`app/tools/registry.py`):

- **Catalog tools** (~60) are what you see as buttons — summarize, analyze dataset,
  character analysis, generate flashcards, KPI extraction, etc. Nearly all of them are
  declarative `PromptToolSpec` entries executed by one generic `PromptTool` class.
- **Primitive tools** (`search_document(s)`, `get_page`, `get_section`, `extract_table`,
  `calculate`, `validate_evidence`, `search_web`, …) are mechanical or narrowly-scoped
  and marked `category="internal"` — reachable by the agent, not rendered as buttons.

Every generation tool retrieves real evidence first, tags each chunk with its
`document_id`/page/section, requires the model to cite using those exact tags, and then
runs the citations back through a heuristic support check before returning them.

Universal (every type): Summarize, Chat, Ask a Question, Explain, Extract Key
Information, Search Documents, Generate Quiz, Find Important Sections, Compare
Documents.

Research paper: Analyze Dataset/Model/Architecture/Methodology, Extract Experimental
Setup, Analyze Metrics, Extract Results, Analyze Limitations, Identify Research Gaps,
Compare Papers, Find Common Techniques, Find Differences, Generate Literature Review,
Generate Research Brief, Web Research.

Literature: Spoiler-Free Summary, Full Summary, Chapter Summary, Character
Analysis/Relationships, Theme/Plot Analysis, Important Events, Motifs/Symbols, Review —
all spoiler-level aware (`none` / `chapter` / `full`).

Study material: Simplified Explanation, Notes, Important Concepts, Definitions, Formula
Extraction, Flashcards, Practice Questions, Important Questions, Question Paper
Generator (configurable marks/difficulty/count/types, optional answer key).

Technical documentation: Architecture Overview, Component/API/Requirements/Dependency
Extraction, Workflow Explanation, Configuration & Security Requirement Extraction,
Implementation Checklist, Technical Summary.

Business report: Executive Summary, KPI/Metric Extraction, Trend Analysis, Risk
Extraction, Action Items, Year-over-Year Comparison, Multi-report Comparison, Business
Brief.

Multi-document comparisons retrieve evidence *per selected document* rather than one
combined query, then return a typed `ComparisonOutput` the frontend renders as a real
table.

## Free LLM gateway & automatic failover

**No paid API is required.** By default, three genuinely free hosted providers are
chained in order — Primary → Secondary → Tertiary:

| Slot | Default provider | Free signup | Notes |
|---|---|---|---|
| Primary | **Groq** | [console.groq.com/keys](https://console.groq.com/keys) | No card required. Fast, generous free-tier rate limits on Llama models. |
| Secondary | **Google Gemini** | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | No card required. Used via Gemini's official OpenAI-compatible endpoint. |
| Tertiary | **OpenRouter** | [openrouter.ai/keys](https://openrouter.ai/keys) | No card required for `:free`-suffixed models. |

The app works with just **one** of these three keys set — more keys just add resilience
against any single provider's free-tier limits. All three (and the optional paid
Anthropic/OpenAI slots) are implemented by one `OpenAICompatibleProvider` class, since
Groq, Gemini's compatibility layer, and OpenRouter all speak the same OpenAI
chat-completions wire format — see `app/llm/openai_compatible_provider.py`.

```
Agent request → ModelGateway.generate(task_type, …)
                        │
                  Try Primary (Groq)
                        │
        ┌───────────────┼────────────────────┐
        │ success        │ quota/auth/connection/5xx   │ bad request (400)
        ▼                ▼                    ▼
   return result   cooldown + try Secondary    log + raise (do NOT fail over --
                        │                       retrying elsewhere won't fix a
                  Try Tertiary if Secondary      malformed prompt, and hiding it
                  also fails                      would hide a real application bug)
                        │
              All exhausted → clean, structured
              AllProvidersUnavailableError (503) --
              never an unexplained 500
```

- A provider that returns a rate-limit/quota error is put on a **cooldown**
  (`LLM_PROVIDER_COOLDOWN_S`, default 90s) — subsequent requests skip it automatically
  instead of wasting requests hammering an already-exhausted free tier.
- A slot whose API key is missing is skipped at startup, silently — **the app starts
  successfully as long as at least one provider is configured**, and even with zero
  configured it still starts (only LLM-dependent requests return a clear error).
- Structured JSON output failures get the same treatment: if a provider's model can't
  produce valid JSON matching the schema after a couple of repair attempts, that
  provider is excluded and the request moves to the next one (`app/llm/structured.py`).
- Free tiers are **not unlimited** — this failover chain is exactly the mitigation for
  that, not a claim that any single provider won't eventually rate-limit you.

Per-task model routing (`app/llm/router.py`) sends classification/extraction/query-
rewrite to each provider's "fast" model and comparison/report-generation to its
"strong" model, independently configurable per slot.

## Vector store abstraction

`app/vectorstores/base.py::VectorStore` has two implementations — `ChromaStore` (local,
default, zero setup) and `QdrantStore` (embedded local, self-hosted, or managed Qdrant
Cloud — all free, same code path, controlled by `QDRANT_URL`). Both use **one collection
per workspace**, and every document within a workspace is additive and filterable by
`document_id` — uploading a new document never clears existing embeddings.

## Evaluation & observability

`GET /api/evaluation/summary` aggregates real recorded rows — nothing here is
hard-coded or randomly generated, and if there's no data yet it says so:

- **Retrieval relevance** — mean similarity score of evidence actually retrieved.
- **Answer faithfulness** — embedding-similarity proxy between the answer and its
  best-matching retrieved evidence (a heuristic, documented as such).
- **Citation accuracy** — share of citations whose excerpt was actually found in the
  source chunk it claims to come from.
- **Primary vs. fallback request ratio, per-provider breakdown** — aggregated from
  `model_usage`, which logs every single provider call, success or failure.

## Dark mode

A full light/dark theme, toggled from the sidebar (persisted via `next-themes`,
respects system preference on first visit, no flash-of-incorrect-theme on reload). Every
color in the app is a semantic CSS variable (`--color-ink`, `--color-paper`,
`--color-cobalt`, …) redefined per theme in `app/globals.css` and consumed through
Tailwind tokens — no component hard-codes a hex value, so light and dark stay in sync
automatically as the UI evolves.

## Setup

### Prerequisites

- Python 3.11+ (project targets 3.12)
- Node.js 18.18+
- PostgreSQL 14+ (or use `docker compose up postgres`; SQLite also works for a
  zero-setup trial — see below)
- At least one free API key: [Groq](https://console.groq.com/keys),
  [Gemini](https://aistudio.google.com/apikey), and/or
  [OpenRouter](https://openrouter.ai/keys)

### 1. Database

```bash
docker compose up -d postgres
```

This creates both the `insightpdf` role and database automatically. To do it by hand
against an existing Postgres instance instead:

```bash
psql postgres -c "CREATE ROLE insightpdf WITH LOGIN PASSWORD 'insightpdf';"
psql postgres -c "CREATE DATABASE insightpdf OWNER insightpdf;"
```

Prefer a fully local trial with zero services? Set
`DATABASE_URL=sqlite:///./insightpdf.db` in `backend/.env` instead and skip this step.

### 2. Qdrant (optional)

Only needed if you set `VECTOR_STORE=qdrant` and want a server instead of the default
embedded local mode (which needs nothing running at all):

```bash
docker compose --profile qdrant up -d qdrant
```

### 3. Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # add at least one of GROQ_API_KEY / GEMINI_API_KEY / OPENROUTER_API_KEY
uvicorn app.main:app --reload --port 8000
```

The first request that needs embeddings will download the sentence-transformers model
(~90MB) — this requires outbound internet access once; it's cached afterwards.

### 4. Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local   # NEXT_PUBLIC_API_URL=http://localhost:8000/api
npm run dev
```

Open `http://localhost:3000`.

## Environment variables

See `backend/.env.example` for the full, commented list. The essentials:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Postgres (recommended) or SQLite connection string |
| `VECTOR_STORE` | `chroma` (default) or `qdrant` |
| `PRIMARY_LLM_PROVIDER` / `SECONDARY_LLM_PROVIDER` / `TERTIARY_LLM_PROVIDER` | `groq` \| `gemini` \| `openrouter` \| `anthropic` \| `openai` |
| `GROQ_API_KEY` / `GEMINI_API_KEY` / `OPENROUTER_API_KEY` | Free provider keys — set at least one |
| `LLM_PROVIDER_COOLDOWN_S` | How long a rate-limited provider is skipped before retrying |
| `TAVILY_API_KEY` | Optional — enables the Web Research tool |
| `RETRIEVAL_MIN_RELEVANCE`, `RETRIEVAL_MAX_REFINE_ATTEMPTS` | Agentic RAG loop tuning |

Never commit a filled-in `.env`. Both `backend/.env.example` and
`frontend/.env.local.example` contain no real secrets.

## Example workflows

**Research workspace.** Upload `paper1.pdf`–`paper4.pdf`. Each gets classified and
ingested independently — if one fails, the other three still become READY. Open one,
click *Analyze Dataset* or *Extract Results* — grounded, cited. Go to **Compare**,
select all four, run *Compare Papers* for a structured comparison table. Ask the
workspace chat "which paper would be easiest to reproduce with limited compute?" — it
retrieves across all four and reasons over the evidence.

**Study workspace.** Upload `Light.pdf`, `Human Eye.pdf`, `Electricity.pdf`. Use
*Question Paper Generator* with `total_marks=30`, pick a difficulty and question count,
optionally include an answer key — every question is grounded in the uploaded chapters.

**A failed upload.** Upload five PDFs where one is corrupted. Four become READY; the
corrupted one shows a **FAILED** card with the specific reason and **Retry**/**Delete**
buttons, always visible — never a dead end, and never blocking the other four.

## Project structure

```
InsightPDF-main/
├── backend/            FastAPI + LangGraph + SQLAlchemy application (see above)
├── frontend/            Next.js 14 (App Router) + TypeScript + Tailwind, light/dark
├── docker-compose.yml    Postgres (+ optional Qdrant) for local dev
└── README.md             You are here
```

## Troubleshooting

**"No LLM provider is configured"** — Set at least one of `GROQ_API_KEY`,
`GEMINI_API_KEY`, or `OPENROUTER_API_KEY` in `backend/.env` and restart the backend. The
app starts fine without one; only chat/tool requests need it.

**A provider keeps hitting rate limits** — Check `/models` in the UI: a rate-limited
provider shows "Rate-limited" with a cooldown countdown and is skipped automatically
until it expires. Add a second/third provider key to reduce how often this matters, or
increase `LLM_PROVIDER_COOLDOWN_S` if you'd rather wait longer between retries.

**"All configured free providers are temporarily unavailable"** — Every configured slot
failed (or is cooling down) for this request. Check `/models` for which ones and why;
this is usually transient. If it persists, verify the relevant API key is still valid.

**`FATAL: role "insightpdf" does not exist`** — Postgres wasn't set up. Run
`docker compose up -d postgres` (creates the role/database for you), or run the two
`psql` commands in [Setup](#1-database) against your own Postgres instance.

**Qdrant unavailable** — If `VECTOR_STORE=qdrant` and `QDRANT_URL` points at nothing
running, either start it (`docker compose --profile qdrant up -d qdrant`) or unset
`QDRANT_URL` to fall back to the embedded local mode (no server required).

**Embedding model won't download** — `sentence-transformers/all-MiniLM-L6-v2` downloads
from Hugging Face on first use and needs outbound internet access once; after that it's
cached locally and works offline. If you're behind a firewall, pre-download it on a
machine with access and copy the Hugging Face cache directory over.

**PDF parsing errors** — A document that fails with "no extractable text" is likely a
scanned/image-only PDF (OCR isn't implemented); one that fails with "could not open as a
PDF" is corrupted or password-protected. Either way, **Retry** and **Delete** are always
available on the document card — a bad file never blocks the rest of your workspace.

**Port conflicts** — Backend defaults to `:8000`, frontend to `:3000`, Postgres to
`:5432`, Qdrant to `:6333`. Override with `uvicorn app.main:app --port <n>` /
`PORT=<n> npm run dev` / editing `docker-compose.yml`'s port mappings respectively, and
update `NEXT_PUBLIC_API_URL` / `DATABASE_URL` / `QDRANT_URL` to match.

## Known limitations & honest scoping notes

- **Background jobs run via FastAPI `BackgroundTasks`**, not Celery/RQ — fine for a
  single-instance deployment; a multi-worker production deployment should move
  ingestion to a real task queue.
- **`Base.metadata.create_all()` instead of Alembic migrations** — fine for getting
  started; introduce Alembic before the schema needs its first real migration.
- **Tool selection uses constrained structured output, not native function-calling** —
  a deliberate choice given inconsistent free-tier tool-calling support; see
  [Agentic RAG](#agentic-rag).
- **OpenRouter's free-model roster changes frequently.** The defaults
  (`TERTIARY_LLM_MODEL_FAST`/`STRONG`) were current as of this writing; if OpenRouter
  ever returns "model not found," the gateway treats it as a normal
  provider-unavailable failure (fails over / logs it) rather than crashing — but you
  should still update the model id from [openrouter.ai/models](https://openrouter.ai/models)
  (filtered to Free) when convenient.
- **Chat progress streaming is coarse-grained** (LangGraph node-level progress via SSE —
  "Searching documents," "Generating answer") not per-token streaming of the final
  answer, since token streaming needs each `LLMProvider` to support streamed
  completions, which the current interface doesn't expose yet.
- **Frontend dependency versions are pinned to Next.js 14.2.35 / React 18.3 / Tailwind
  3.4** rather than the latest majors, a deliberate choice to ship code whose exact API
  surface could be verified rather than guessed at. `npm audit` shows one remaining
  advisory cluster in Next 14.2.35 related to Server Actions/Edge rewrites/custom-server
  SSRF — this app uses none of those (no Server Actions, no custom server, no
  rewrites).
- **No authentication** — this build is single-tenant-per-deployment. Adding auth means
  a `users` table, a session/JWT layer, and scoping workspace/document queries by
  owner — the service layer is already structured to make that additive.
- **SQLite works but Postgres is recommended.** SQLite only allows one writer at a
  time; the app checkpoints commits at safe points and enables WAL mode + a busy-timeout
  + `PRAGMA foreign_keys=ON` as a safety net. Postgres has none of this constraint and
  correctly enforces every `ON DELETE CASCADE` at the database level.
