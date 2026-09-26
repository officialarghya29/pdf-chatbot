<p align="center">
  <img src="frontend/public/favicon.svg" width="80" alt="Unfold logo" />
</p>

<h1 align="center">
  <strong>Unfold</strong><br/>
  <sub>Unfold the knowledge hidden inside your documents.</sub>
</h1>

<p align="center">
  <img src="frontend/public/screenshot.png" width="600" alt="Unfold UI screenshot" />
</p>

<p align="center">
  A full-stack <strong>retrieval-augmented PDF chat</strong> application.<br/>
  Upload a document, ask questions in plain language, and get streamed answers<br/>
  with citations mapped to the exact pages they came from.
</p>

<p align="center">
  <a href="#what-unfold-is">Overview</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#features">Features</a> ·
  <a href="#api-reference">API</a> ·
  <a href="#configuration">Config</a> ·
  <a href="#testing">Testing</a> ·
  <a href="#deployment">Deployment</a> ·
  <a href="#design-notes">Design notes</a> ·
  <a href="#troubleshooting">Troubleshooting</a> ·
  <a href="#author">Author</a>
</p>

---

## What Unfold is

Unfold is a self-hostable application for asking questions about PDF documents.
It combines three ordinary pieces — a PDF text extractor, a vector index, and a
chat model — into one product with a working UI, persistent sessions, and
page-level citations.

You give it a document. It breaks the document into passages, embeds those
passages, and stores them in a local vector index. When you ask a question, it
retrieves the passages most similar to your question, hands them to the model as
context, and streams the answer back while keeping track of which passages were
used. The `[1]`, `[2]` markers in an answer are not decorative — each one points
at a page you can open in the source panel.

### What it is not

- It is not a general-purpose chatbot. It only answers from the document you
  uploaded, and it will say so when the document does not contain an answer.
- It is not a document manager. One PDF per session by design.
- It is not tuned for very large corpora. FAISS `IndexFlatIP` is a brute-force
  index; it is fast for hundreds of thousands of vectors and simple to reason
  about, but it is not a sharded production search cluster.

### Why it exists

Reading a long PDF to find one fact is a slow, linear process. Search finds
substrings, not meaning. Unfold answers the actual question and shows you where
the answer came from, which is the part most naive RAG demos skip.

---

## How it works

### Pipeline

```
UPLOAD
  PDF ──► pypdf: extract text page by page
      ──► clean:  strip repeating headers/footers, repair hyphenated line breaks
      ──► chunk:  sliding window of words (size 1000, overlap 200)
      ──► embed:  text-embedding-3-small, batched
      ──► index:  FAISS IndexFlatIP over L2-normalized vectors (cosine)
      ──► store:  session meta + index persisted to disk

QUERY
  question ──► embed the query
           ──► retrieve: FAISS candidate pool for dense similarity
           ──► score:   blend dense similarity with a BM25 lexical score
           ──► select:  MMR over the top-5 so they are not near-duplicates
           ──► build prompt: system rules + retrieved context + recent history + question
           ──► stream the completion token by token over SSE
           ──► parse [n] markers from the final answer, resolve to page numbers
           ──► persist the turn so history survives a restart
```

### Request flow

```
  Browser                  Vite dev server            FastAPI backend
     │                            │                          │
     │  GET /api/health ─────────►│─────────────────────────►│
     │◄── {name, version, llm_configured: bool} ─────────────│
     │                            │                          │
     │  drag & drop a PDF         │                          │
     │  POST /api/upload (XHR, progress) ───────────────────►│
     │                            │                    extract │
     │                            │                    chunk   │
     │                            │                    embed   │
     │                            │                    index   │
     │◄── {session_id, title, pages, chunks} ────────────────│
     │                            │                          │
     │  POST /api/ask (SSE) ─────────────────────────────────►│
     │◄── event: start  {sources:[...]} ─────────────────────│
     │◄── event: delta  {text:"..."}  (repeated) ────────────│
     │◄── event: done   {answer, citations} ─────────────────│
```

Streaming matters here: a four-paragraph answer takes several seconds to
generate, and waiting for the whole thing feels broken. Tokens are rendered as
they arrive. If the stream is cut without a `done` event — a proxy timeout, a
server restart — the client detects the truncation and surfaces a retryable
error instead of spinning forever.

---

## Quick start

### Requirements

- Python 3.10 or newer
- Node.js 18 or newer
- An API key for OpenAI, or any OpenAI-compatible provider (see
  [Configuration](#configuration))

### Install

```bash
git clone https://github.com/officialarghya29/pdf-chatbot.git
cd pdf-chatbot

# Backend
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then put your OPENAI_API_KEY in it
cd ..

# Frontend
cd frontend
npm install
cd ..
```

### Run

```bash
# Terminal 1 — backend on :8000
cd backend && uvicorn main:app --reload --port 8000

# Terminal 2 — frontend on :5173
cd frontend && npm run dev
```

Open `http://localhost:5173`, upload a PDF, and start asking questions. In
development Vite proxies `/api/*` to the backend, so there is nothing else to
configure.

### Or use the helper script

```bash
cp .env.example.docker .env     # set OPENAI_API_KEY
./dev.sh                        # starts both servers, Ctrl+C stops both
```

---

## Features

### Core

| Feature | Detail |
|---|---|
| PDF ingestion | Page-by-page text extraction; strips repeating headers/footers; repairs hyphenated line breaks; detects encrypted and image-only (scanned) PDFs and reports them clearly |
| Chunking | Sliding-window word chunking so passages never lose their page number |
| Hybrid retrieval | FAISS `IndexFlatIP` cosine search over normalized embeddings, blended with a hand-written BM25 lexical score so exact terms (IDs, names, numbers) survive |
| MMR diversification | Greedy Maximal Marginal Relevance over the top-k, so five retrieved passages are five pieces of evidence rather than five paraphrases |
| Streaming answers | Server-Sent Events; answers render token by token |
| Inline citations | `[n]` markers parsed from the answer and resolved to the pages they came from |
| Persistent sessions | Indexes, metadata, and chat history are written to disk and reload on restart |
| Provider-agnostic | Any OpenAI-compatible endpoint — OpenAI, Azure, OpenRouter, Groq, local Ollama |
| Concurrency-safe | Session store is lock-protected; PDF parsing and embedding run in worker threads so indexing never blocks the event loop |

### Interface

| Feature | Detail |
|---|---|
| Session sidebar | Create, switch, clear, and delete sessions; document count per session |
| Source panel | Expandable per-answer list of retrieved passages with page chips |
| Copy answer | One-click copy on any assistant message |
| Jump to bottom | Appears when you scroll up; disappears at the bottom. Auto-scroll only follows while you are already near the bottom |
| Upload progress | Drag-and-drop with a percentage progress bar |
| Clear history | Wipes the conversation but keeps the document indexed |
| API key warning | An amber banner when the backend reports no key configured, so the failure is visible before the first upload |
| Error boundary | A render crash shows a recoverable screen instead of a blank page |
| Responsive | Hamburger sidebar on small screens |
| Compression | Gzip on API responses |
| Request logging | Method, path, status, and latency for every request |

---

## API reference

All routes are prefixed with `/api`.

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/health` | Status, version, and whether an LLM key is configured |
| `POST` | `/api/upload` | Upload a PDF (multipart) → creates a session |
| `GET` | `/api/sessions` | List sessions, newest first |
| `GET` | `/api/sessions/{id}` | One session's summary (title, pages, chunks) |
| `GET` | `/api/sessions/{id}/messages` | Chat history for a session |
| `DELETE` | `/api/sessions/{id}` | Delete the session, its index, and its history |
| `DELETE` | `/api/sessions/{id}/messages` | Clear the chat history, keep the document |
| `GET` | `/api/sessions/{id}/export` | Download the conversation as a Markdown transcript |
| `POST` | `/api/ask` | Ask a question → SSE stream |

### SSE stream format

```
data: {"type":"start","sources":[{"page":3,"snippet":"..."}]}

data: {"type":"delta","text":"The authors"}

data: {"type":"delta","text":" report a 12% improvement"}

data: {"type":"done","answer":"The authors report a 12% improvement.","citations":[{"n":1,"page":3}]}
```

An error mid-stream is delivered as `{"type":"error","message":"..."}` so the
client can show it without the HTTP request failing.

### Example

```bash
# Upload a document
curl -X POST http://localhost:8000/api/upload -F "file=@paper.pdf"
# → {"session_id":"a1b2c3d4e5f6","title":"…","pages":12,"chunks":15}

# Ask about it
curl -N -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"session_id":"a1b2c3d4e5f6","query":"What are the key findings?"}'

# Clear the conversation
curl -X DELETE http://localhost:8000/api/sessions/a1b2c3d4e5f6/messages

# Remove the document
curl -X DELETE http://localhost:8000/api/sessions/a1b2c3d4e5f6
```

---

## Configuration

All settings live in `backend/.env`. See `backend/.env.example`.

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** Key for your provider |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Compatible endpoint |
| `CHAT_MODEL` | `gpt-4o-mini` | Chat completion model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `TEMPERATURE` | `0.3` | Generation randomness |
| `CHUNK_SIZE` | `1000` | Words per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `TOP_K` | `5` | Passages retrieved per question |
| `MAX_HISTORY` | `10` | Conversation turns sent to the model |
| `HYBRID_SEARCH` | `true` | Blend dense and lexical retrieval; `false` = pure vector search |
| `HYBRID_ALPHA` | `0.35` | Lexical share of the blended score (0.0–1.0) |
| `MMR_LAMBDA` | `0.7` | Diversification strength; `1.0` = pure relevance |
| `MAX_UPLOAD_MB` | `25` | Upload size limit |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |

### Other providers

```bash
# OpenRouter
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=sk-or-...
CHAT_MODEL=openai/gpt-4o-mini

# Groq
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_API_KEY=gsk_...
CHAT_MODEL=llama-3.1-70b-versatile

# Local Ollama
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
CHAT_MODEL=llama3.1
EMBEDDING_MODEL=nomic-embed-text
```

If you use a remote provider, remember to set `CORS_ORIGINS` to the origin your
frontend is actually served from rather than `*`.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  Frontend — Vite + React 18 + TypeScript + Tailwind                │
│                                                                    │
│  App.tsx  ─ orchestrates sessions, streaming, unread state         │
│    ├── Sidebar       sessions, clear, delete, mobile drawer        │
│    ├── EmptyState    landing + suggested questions                 │
│    ├── MessageBubble markdown, citations, copy, thinking state     │
│    ├── SourcePanel   retrieved passages, page chips                │
│    ├── Composer      drag-drop upload + question input             │
│    ├── StatusBar     health + connection state                     │
│    ├── ApiWarning    missing-key banner                            │
│    └── ErrorBoundary render-crash recovery                         │
│                                                                    │
│  lib/api.ts — typed client + SSE parser (handles chunk splitting)  │
└─────────────────────────────┬──────────────────────────────────────┘
                              │ /api/*  (JSON, multipart, SSE)
┌─────────────────────────────┴──────────────────────────────────────┐
│  Backend — FastAPI + uvicorn                                       │
│                                                                    │
│  main.py     routes, SSE generator, middleware, error mapping      │
│  ingest.py   pypdf extraction, cleaning, chunking                  │
│  sessions.py thread-safe session store, FAISS index, persistence   │
│  llm.py      OpenAI client: embed, chat, stream, error mapping     │
│  schemas.py  pydantic request/response models                      │
│  config.py   env-driven settings                                   │
└────────────────────────────────────────────────────────────────────┘
```

### Stack and rationale

| Layer | Choice | Why |
|---|---|---|
| PDF parsing | `pypdf` | Pure Python, no system dependencies, handles encrypted files |
| Chunking | Hand-written | ~40 lines; keeps page metadata exact and is trivial to tune |
| Embeddings | `text-embedding-3-small` | Good retrieval quality at low cost |
| Vector index | FAISS `IndexFlatIP` | Exact cosine search with no server to run |
| Model | `gpt-4o-mini` (default) | Cheap and adequate for grounded, context-supplied answers |
| Backend | FastAPI + uvicorn | Native async support, which SSE streaming needs |
| Frontend | React + TypeScript + Tailwind | Typed end to end; no CSS framework runtime |
| Deployment | Docker, nginx, Gunicorn/uvicorn | One-command local run; either container or split cloud |

---

## Testing

128 automated checks. They run entirely offline — the language model is mocked,
so the suites cost nothing and are deterministic.

```bash
# Backend — 104 checks
python backend/smoke_test.py       # 20  end-to-end API flow
python backend/advanced_test.py    # 29  abuse, concurrency, eviction, fuzzing
python backend/deepscan_test.py    # 22  persistence, logging, stream failure, edges
python backend/retrieval_test.py   # 33  BM25, hybrid blending, MMR, export endpoint

# Frontend — 24 checks
cd frontend && npm test            # vitest + Testing Library + jsdom

# Type check and production build
cd frontend && npm run build       # strict tsc, then vite build
```

| Area | Checks | Covers |
|---|---|---|
| Health and sessions | 8 | CRUD, listing, history |
| Upload validation | 12 | Non-PDF, corrupt, oversized, encrypted, image-only, empty, very long filenames |
| Chat and streaming | 15 | SSE event order, citations, model failure, mid-stream failure, partial-answer persistence |
| Persistence | 6 | Disk round-trip, corrupt metadata tolerance, eviction |
| Concurrency | 4 | Parallel asks; ask + clear + history simultaneously |
| Edge cases | 16 | Path separators, hostile strings, whitespace-only queries, unicode, extra fields |
| Components | 10 | MessageBubble, Composer, Sidebar, copy button, thinking state |
| Retrieval engine | 33 | BM25 scoring, hybrid blending, MMR diversity, degenerate and empty indices |
| SSE parsing | 7 | Events split across chunk boundaries, truncation, malformed events, abort |

A few checks exercise more than one of these areas at once, so the per-area
counts are a rough guide and do not add up to exactly 128.

Several real bugs were found by these tests and fixed: a whitespace-only query
that passed validation, a truncated stream that left the UI spinning forever,
and a partial answer that was lost when the client disconnected mid-stream.

---

## Deployment

### Docker

```bash
cp .env.example.docker .env       # set OPENAI_API_KEY
docker compose up --build         # → http://localhost:8080
```

Compose runs the FastAPI backend and an nginx-served frontend with a shared
volume for session data. The nginx config disables proxy buffering on `/api`
so SSE streams arrive incrementally.

### Split cloud

| Part | Platform | Config |
|---|---|---|
| Backend | Render (or any container host) | `render.yaml` |
| Frontend | Vercel (or any static host) | `vercel.json` |

`vercel.json` rewrites `/api/*` to the backend origin, so streaming works
through the Vercel edge without CORS setup. Note that persistent session data
needs a disk — Render's free tier has no persistent disk, so sessions reset on
redeploy unless you attach one on a paid plan or point storage at a volume.

### Local

```bash
./dev.sh
```

---

## Design notes

Two decisions shaped this project.

**Retrieval before generation, always.** The model never answers from its own
memory. Every answer is produced from passages retrieved out of your document,
and the citations make that auditable. When retrieval finds nothing relevant,
the prompt instructs the model to say the document does not cover the question
rather than invent an answer.

**No orchestration framework.** The RAG loop here is small enough to write
directly against the provider's SDK: embed, search, build a prompt, stream.
Frameworks like LangChain are useful for rapidly wiring together many
integrations, but for a single well-understood pipeline they add a dependency
graph and an indirection layer that make streaming and citation tracking harder,
not easier. This project deliberately stays at the SDK level — eleven backend
dependencies, every step readable in `ingest.py`, `sessions.py`, and `llm.py`.

That is a trade-off, not a universal rule. If you need dozens of loaders,
multi-hop agents, or managed tracing, a framework likely earns its keep.

### Honest limitations

- One PDF per session; no cross-document retrieval.
- `IndexFlatIP` is brute force. It is fine to tens of thousands of vectors and
  will need an IVF/HNSW index beyond that.
- Text extraction quality follows the source PDF. Scanned documents without an
  OCR layer are rejected rather than silently returning nothing.
- Retrieval is single-pass: no query rewriting, no cross-encoder reranker, no multi-hop search. Hybrid BM25 blending and MMR diversification are built in.
- Sessions are stored on local disk; running multiple backend replicas against
  a shared store is not supported.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `OPENAI_API_KEY is not configured` | No key in `backend/.env` | Add the key and restart the backend |
| `Invalid or missing OPENAI_API_KEY` | Wrong or revoked key | Check the key with your provider |
| `Rate limit / quota exceeded` | Provider quota | Wait, or raise your plan |
| `Could not reach the LLM provider` | Network or wrong `OPENAI_BASE_URL` | Check connectivity and the base URL |
| `Only PDF files are supported` | Non-PDF upload | Upload a `.pdf` |
| `No extractable text found` | Scanned / image-only PDF | Use a text-based PDF, or OCR it first |
| `This PDF is password protected` | Encrypted PDF | Remove the password before uploading |
| `File too large` | Over `MAX_UPLOAD_MB` | Compress or split the file |
| `Session not found` | Deleted, or data was not persisted | Re-upload; attach a volume when deploying |
| CORS error in the browser | Backend down or origin not allowed | Start the backend; set `CORS_ORIGINS` |
| "Backend offline" in the UI | Backend not running | `cd backend && uvicorn main:app --reload --port 8000` |
| Blank page | Render crash | The error boundary shows a recovery screen; check the browser console |

---

## Project structure

```
pdf-chatbot/
├── backend/
│   ├── main.py               FastAPI app, routes, SSE streaming, middleware
│   ├── config.py             Settings loaded from environment
│   ├── ingest.py             PDF extraction, cleaning, chunking
│   ├── llm.py                Provider client: embeddings, chat, streaming
│   ├── sessions.py           Session store, FAISS index, persistence
│   ├── schemas.py            Request/response models
│   ├── smoke_test.py         20 end-to-end API tests
│   ├── advanced_test.py      29 abuse / concurrency / edge-case tests│   ├── deepscan_test.py     22 persistence / logging / failure tests
│   ├── retrieval_test.py    33 retrieval engine / export tests
│   ├── requirements.txt      Python dependencies
│   ├── .env.example          Environment template
│   └── Dockerfile            Backend image
├── frontend/
│   ├── src/
│   │   ├── App.tsx           Session state and streaming orchestration
│   │   ├── main.tsx          Entry point
│   │   ├── index.css         Tailwind theme
│   │   ├── components/       UI components (see Architecture)
│   │   ├── lib/
│   │   │   ├── api.ts        Typed API client and SSE parser
│   │   │   └── utils.ts      Formatting helpers
│   │   └── test/setup.ts     Vitest setup
│   ├── public/               favicon, screenshot
│   ├── index.html
│   ├── vite.config.ts        Dev proxy + test config
│   ├── tsconfig.json         Strict TypeScript
│   ├── nginx.conf            SPA fallback + SSE-safe proxy
│   └── Dockerfile            Multi-stage build → nginx
├── docker-compose.yml        Full stack, persistent volume
├── vercel.json               Frontend deploy + API rewrite
├── render.yaml               Backend blueprint
├── dev.sh                    Start both servers locally
├── .env.example.docker       Docker environment template
├── LICENSE
└── README.md
```

---

## Author

**officialarghya29** — [github.com/officialarghya29](https://github.com/officialarghya29)

Designed and built as a complete, deployable RAG application: backend,
frontend, tests, and deployment configuration.

If Unfold is useful to you, a star on the repository is appreciated. Questions
and bug reports are welcome through GitHub Issues.

---

## Contributing

1. Fork the repository and branch from `main`.
2. Make your change.
3. Run the tests: `python backend/smoke_test.py`, then `cd frontend && npm test`.
4. Open a pull request describing what changed and why.

Guidelines: keep routes thin and logic in the service modules; add a
regression test with every bug fix; TypeScript stays in strict mode.

---

## License

MIT — see [LICENSE](LICENSE). © 2026 officialarghya29.

---

<p align="center">
  <sub>Unfold v3.0.0 · 128 tests · 11 backend dependencies · no orchestration framework</sub>
</p>
