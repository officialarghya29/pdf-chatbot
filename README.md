# NeoChat · AI PDF Chatbot

A futuristic, production-grade chat-with-your-PDF app.

- **Frontend** — React 18 + TypeScript + Vite + Tailwind CSS, glassmorphism dark UI,
  streaming answers, drag-&-drop uploads, source citations, session history.
- **Backend** — FastAPI + OpenAI-compatible API, FAISS vector search,
  page-aware chunking, SSE token streaming, disk-persisted sessions.

---

## Quick start

### 1. Backend

```bash
cd backend
python3 -m venv venv
venv/bin/pip install -r requirements.txt   # Windows: venv\Scripts\pip install -r requirements.txt

cp .env.example .env                       # then edit .env and set OPENAI_API_KEY
venv/bin/uvicorn main:app --reload --port 8000
```

API docs open at <http://localhost:8000/docs>.

> Works with any OpenAI-compatible endpoint — set `OPENAI_BASE_URL` in `.env`.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173  (proxies /api → :8000)
```

### One-command dev (optional)

```bash
./dev.sh           # starts backend + frontend together
```

---

## Features

| Area | What you get |
|---|---|
| Ingestion | PDF text extraction, header/footer cleaning, metadata titles, page-aware chunking |
| Retrieval | FAISS cosine similarity, normalized embeddings, top-k configurable |
| Chat | SSE token streaming, conversation memory, markdown answers, inline citations `[n] → page` |
| Sessions | Multiple documents, persistent history across restarts, delete, auto-eviction |
| Safety | File-type/magic-byte/size validation, friendly LLM error mapping, CORS config |
| UX | Glass UI, animated hero, drag-drop with progress %, stop-generation, offline detection |

## Configuration (backend/.env)

| Var | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | Required. Your secret key |
| `OPENAI_BASE_URL` | OpenAI | Any compatible endpoint |
| `CHAT_MODEL` | `gpt-4o-mini` | Chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1000 / 200 | RAG chunking |
| `TOP_K` | 5 | Chunks retrieved per question |
| `MAX_UPLOAD_MB` | 25 | Upload limit |
| `CORS_ORIGINS` | localhost origins | Comma-separated allow-list |

## API

| Method | Route | Description |
|---|---|---|
| GET | `/api/health` | Health + config status |
| POST | `/api/upload` | Upload PDF → creates session (multipart `file`) |
| GET | `/api/sessions` | List sessions |
| GET | `/api/sessions/{id}` | Session summary |
| GET | `/api/sessions/{id}/messages` | Chat history |
| DELETE | `/api/sessions/{id}` | Delete session + vectors |
| POST | `/api/ask` | Ask a question → **SSE stream** (`start`→`delta`…→`done`) |

## Testing

```bash
backend/venv/bin/python backend/smoke_test.py     # 20-check end-to-end API flow (mocked LLM)
backend/venv/bin/python backend/advanced_test.py  # 29-check suite: abuse, unicode, concurrency,
                                                  #   eviction, corrupt storage, chunker fuzzing
cd frontend && npm test                           # 10 vitest unit tests (SSE parser, utils)
cd frontend && npm run build                      # strict typecheck + production build
```

All 59 checks pass. Tests run fully offline — the LLM layer is mocked.

## Project layout

```
backend/
  main.py        FastAPI app & routes
  config.py      Env-driven settings
  ingest.py      PDF extraction & chunking
  llm.py         OpenAI client (chat/embed/stream)
  sessions.py    Session store, FAISS search, persistence
  schemas.py     Pydantic models
  smoke_test.py    End-to-end API tests
  advanced_test.py Abuse/concurrency/persistence test suite
frontend/
  src/components/  Sidebar, Composer, MessageBubble, SourcePanel, …
  src/lib/api.ts   Typed API client + SSE parser + XHR upload progress
```
