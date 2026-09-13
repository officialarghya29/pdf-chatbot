<p align="center">
  <img src="frontend/public/favicon.svg" width="96" alt="Unfold logo" />
</p>

<h1 align="center">
  <strong>Unfold</strong><br/>
  <sub>Unfold the knowledge hidden inside your documents.</sub>
</h1>

<p align="center">
  <img src="frontend/public/screenshot.png" width="500" alt="Unfold UI screenshot" />
</p>

<p align="center">
  <strong>AI-powered PDF chat</strong> — upload any document, ask questions in natural language,
  get answers with inline citations mapped to exact pages.
  Built with FastAPI + FAISS + OpenAI + React 18 + Tailwind CSS.
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="#features">Features</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#api">API</a> ·
  <a href="#deployment">Deployment</a> ·
  <a href="#comparison">Why Unfold?</a>
</p>

---

## Quick Start

```bash
# 1. Clone and enter
git clone https://github.com/officialarghya29/pdf-chatbot.git
cd pdf-chatbot

# 2. Backend
cd backend
python3 -m venv venv
venv/bin/pip install -r requirements.txt
cp .env.example .env          # ← paste your OPENAI_API_KEY here
venv/bin/uvicorn main:app --reload --port 8000

# 3. Frontend (new terminal)
cd frontend
npm install
npm run dev                   # → http://localhost:5173
```

Or use the one-command shortcut:
```bash
cp .env.example.docker .env   # set OPENAI_API_KEY
./dev.sh                      # starts both
```

---

## Features

| Capability | Detail |
|---|---|
| **PDF Ingestion** | Page-aware text extraction via `pypdf`, header/footer cleanup, hyphenation repair, encrypted-PDF detection |
| **Smart Chunking** | Sliding-window word chunking (configurable 1000/200), per-page metadata preserved |
| **Vector Search** | FAISS `IndexFlatIP` with L2-normalized embeddings → cosine similarity, top-k configurable |
| **Streaming Chat** | SSE token streaming: answers appear in real time as the model generates them |
| **Inline Citations** | `[1]`, `[2]` markers in the answer map to exact pages via expandable source chips |
| **Copy Button** | One-click copy on every assistant message with visual feedback |
| **Jump to Bottom** | Floating button appears when you scroll up, auto-scrolls on new tokens |
| **Clear History** | Eraser icon per session clears chat history without deleting the document |
| **Mobile Sidebar** | Hamburger menu + slide-in sidebar on mobile with backdrop overlay |
| **Multi-Document Sessions** | Upload multiple PDFs per session, cross-document Q&A |
| **Persistence** | FAISS indices + chat history + metadata survive server restarts (disk-backed) |
| **Concurrency-Safe** | Thread-safe session store, heavy work on worker threads (never blocks the event loop) |
| **Gzip Compression** | Automatic response compression for all API endpoints |
| **Request Logging** | Every request logged with method, path, status code, and latency |
| **Error Boundary** | React `ErrorBoundary` catches render crashes; backend friendly-maps every OpenAI error |
| **API Key Warning** | Dismissible banner when OPENAI_API_KEY is missing, with setup instructions |
| **Docker Ready** | `docker compose up --build` → full stack on `:8080` |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Frontend (Vite + React 18 + TypeScript + Tailwind)                │
│                                                                     │
│  ┌────────────┐  ┌──────────┐  ┌────────────┐  ┌──────────────┐   │
│  │ Sidebar     │  │ Chat     │  │ Source     │  │ Composer     │   │
│  │ (sessions)  │  │ (stream) │  │ Panel      │  │ (upload+ask) │   │
│  └──────┬─────┘  └────┬─────┘  └────────────┘  └──────┬───────┘   │
│         │              │                                │            │
│         └──────────────┼────────────────────────────────┘            │
│                        │ SSE events + XHR upload                    │
└────────────────────────┼────────────────────────────────────────────┘
                         │  /api/*
┌────────────────────────┼────────────────────────────────────────────┐
│  Backend (FastAPI + uvicorn)                                        │
│                        │                                            │
│  ┌─────────────────────┴──────────────────────────────────────┐    │
│  │ Router: /api/upload  /api/ask  /api/sessions  /api/health  │    │
│  └────────┬──────────────────┬───────────────────┬────────────┘    │
│           │                  │                   │                   │
│  ┌────────▼─────────┐ ┌─────▼──────────┐ ┌──────▼────────────┐   │
│  │ ingest.py         │ │ sessions.py    │ │ llm.py            │   │
│  │  pypdf → chunks   │ │  FAISS index   │ │  OpenAI client    │   │
│  │  header cleanup   │ │  persistence   │ │  streaming chat   │   │
│  └──────────────────┘ │  prompt builder │ │  embeddings       │   │
│                        └────────────────┘ └───────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

### Data flow — Upload to first answer

```
User drops PDF        Frontend POST /api/upload        Backend
     │                      │                             │
     │  ──────────────────► │  multipart/form-data        │
     │                      │  ─────────────────────────► │
     │                      │                             │  extract_pdf()  → text per page
     │                      │                             │  chunk_text()   → sliding windows
     │                      │                             │  embed_texts()  → FAISS vectors
     │                      │                             │  index.add()    → cosine search
     │                      │  ◄───────────────────────── │  200 + session summary
     │  ◄────────────────── │                             │
     │                      │                             │
User asks question     POST /api/ask (SSE stream)       Backend
     │                      │                             │
     │  ──────────────────► │  {session_id, query}        │
     │                      │  ─────────────────────────► │
     │                      │                             │  search(query)   → top-k chunks
     │                      │                             │  build_messages() → system + history
     │                      │                             │  llm.stream()    → token deltas
     │                      │  ◄─── event: start ──────── │
     │                      │  ◄─── event: delta ──────── │
     │  ◄── "Answer... "   │  ◄─── event: delta ──────── │
     │  ◄── [citation]     │  ◄─── event: done ────────── │
```

---

## API

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/api/health` | Health check + LLM config status |
| `POST` | `/api/upload` | Upload PDF → creates session (multipart) |
| `GET` | `/api/sessions` | List all sessions (newest first) |
| `GET` | `/api/sessions/{id}` | Session summary (title, pages, chunks) |
| `GET` | `/api/sessions/{id}/messages` | Chat history for a session |
| `DELETE` | `/api/sessions/{id}` | Delete session + vector index + history |
| `DELETE` | `/api/sessions/{id}/messages` | Clear chat history (keeps session) |
| `POST` | `/api/ask` | Ask a question → **SSE stream** (`start` → `delta`s → `done`) |

### SSE event format

```json
{"type": "start", "sources": [{"page": 3, "snippet": "..."}]}
{"type": "delta",  "text": "The answer"}
{"type": "delta",  "text": " is 42"}
{"type": "done",   "answer": "The answer is 42", "citations": [{"n": 1, "page": 3}]}
```

---

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required.** Your secret key ([get one](https://platform.openai.com/api-keys)) |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Compatible endpoint (Azure, OpenRouter, Groq) |
| `CHAT_MODEL` | `gpt-4o-mini` | Chat completion model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `1000` | Approximate characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `TOP_K` | `5` | Chunks retrieved per query |
| `MAX_HISTORY` | `10` | Conversation turns sent to the LLM |
| `MAX_UPLOAD_MB` | `25` | Max PDF upload size |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `OPENAI_API_KEY is not configured` | No key in `backend/.env` | Create `backend/.env` with `OPENAI_API_KEY=sk-...` and restart |
| `Invalid or missing OPENAI_API_KEY` | Wrong key or expired | Check your key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `Rate limit / quota exceeded` | Too many requests or free tier limit | Wait or upgrade your OpenAI plan |
| `Could not reach the LLM provider` | Network issue or wrong `OPENAI_BASE_URL` | Check internet connection and base URL |
| `Only PDF files are supported` | Uploaded non-PDF file | Upload a `.pdf` file |
| `No extractable text found` | Scanned/image-only PDF | Upload a text-based PDF (not a scan) |
| `This PDF is password protected` | Encrypted PDF | Remove password protection before uploading |
| `File too large` | PDF exceeds 25 MB | Compress or split the PDF |
| `Session not found` | Session was deleted or server restarted | Upload the PDF again |
| CORS errors in browser | Backend not running or wrong origin | Start backend on :8000, check `CORS_ORIGINS` in `.env` |
| Frontend shows "Backend offline" | Backend not started | Run `cd backend && venv/bin/uvicorn main:app --reload --port 8000` |
| Blank white page | React crash | Check browser console; ErrorBoundary should show recovery UI |

---

## Why Unfold?

### Comparison with alternatives

| Feature | Unfold | Basic chatbots | LangChain demos | Enterprise RAG |
|---------|--------|----------------|-----------------|----------------|
| **Setup time** | `pip install` + env | Minutes | Hours | Weeks |
| **Dependencies** | 7 packages (no langchain) | Varies | 20+ | Dozens |
| **Streaming answers** | ✅ SSE real-time | ❌ Wait for full | ⚠️ Generator | ✅ |
| **Citations → pages** | ✅ Inline `[1]` → page | ❌ None | ⚠️ Manual | ⚠️ Extra work |
| **Multi-document** | ✅ Per-session | ❌ Single | ⚠️ Manual | ✅ |
| **Persistence** | ✅ Disk-backed | ❌ Ephemeral | ❌ Ephemeral | ✅ |
| **Error handling** | ✅ Friendly messages | ❌ Raw traceback | ⚠️ Generic | ✅ |
| **Frontend quality** | ✅ Futuristic UI | ⚠️ Bare minimal | ⚠️ Streamlit | ✅ Custom |
| **Docker deploy** | ✅ `docker compose up` | ❌ Manual | ❌ Manual | ⚠️ K8s |
| **Free deployment** | ✅ Vercel + Render | — | — | Paid only |

### Design philosophy

```
Traditional approach:
  pip install langchain openai chromadb ...    ← 20+ packages
  from langchain import ...                    ← deep coupling
  chain = RetrievalQA.from_chain_type(...)     ← opaque abstraction
  result = chain.run(query)                    ← no streaming, no citations

Unfold approach:
  pip install openai faiss-cpu pypdf fastapi   ← 7 packages, zero abstractions
  client = OpenAI(api_key=...)                 ← direct API, transparent
  index.add(embed_texts(chunks))               ← explicit vector search
  for delta in stream_chat(messages): ...      ← real streaming, full control
```

Unfold avoids framework abstraction layers. Every line of code is readable, debuggable, and replaceable. The result is fewer bugs, faster startup, and an app that actually works.

---

## Testing

```bash
# Backend (50 checks, fully offline — LLM is mocked)
backend/venv/bin/python backend/smoke_test.py      # 20 end-to-end API flow checks
backend/venv/bin/python backend/advanced_test.py    # 29 abuse, concurrency, eviction, fuzzing

# Frontend (19 checks — vitest + Testing Library + jsdom)
cd frontend && npm test                             # utils, SSE parser, component renders

# Full build verification
cd frontend && npm run build                        # strict tsc + production bundle
```

**68 total checks** — all pass offline with no API key.

---

## Deployment

### Option A — Docker (simplest)

```bash
cp .env.example.docker .env   # set OPENAI_API_KEY
docker compose up --build     # → http://localhost:8080
```

### Option B — Split cloud (free tiers)

1. **Backend** → [Render](https://render.com) (Blueprint via `render.yaml`)
2. **Frontend** → [Vercel](https://vercel.com) (imports repo, `vercel.json` rewrites `/api/*` to Render)

### Option C — Local dev

```bash
./dev.sh   # starts backend :8000 + frontend :5173
```

---

## Project structure

```
unfold/
├── backend/
│   ├── main.py          FastAPI app & routes
│   ├── config.py        Env-driven settings (pydantic-settings)
│   ├── ingest.py        PDF extraction, cleaning & chunking
│   ├── llm.py           OpenAI client (chat / embed / stream)
│   ├── sessions.py      Session store, FAISS search, persistence
│   ├── schemas.py       Pydantic request/response models
│   ├── smoke_test.py    20 end-to-end API tests (mocked LLM)
│   ├── advanced_test.py 29 abuse, concurrency & edge-case tests
│   └── Dockerfile       Production container image
├── frontend/
│   ├── src/
│   │   ├── App.tsx                  Root state + streaming orchestration
│   │   ├── main.tsx                 Entry with ErrorBoundary
│   │   ├── index.css                Tailwind theme + glassmorphism
│   │   ├── components/
│   │   │   ├── Sidebar.tsx          Session list + clear history + mobile hamburger
│   │   │   ├── EmptyState.tsx       Animated hero + suggestion cards
│   │   │   ├── MessageBubble.tsx    Markdown + citations + copy button + thinking dots
│   │   │   ├── Composer.tsx         Upload + message input + drag-drop
│   │   │   ├── SourcePanel.tsx      Expandable source chunks
│   │   │   ├── StatusBar.tsx        Health + connection + model info
│   │   │   ├── ApiWarning.tsx       Missing API key banner
│   │   │   └── ErrorBoundary.tsx    Crash recovery UI
│   │   └── lib/
│   │       ├── api.ts              Typed API client + SSE parser
│   │       └── utils.ts            formatBytes, timeAgo, uid
│   ├── *.test.tsx                   19 vitest + Testing Library tests
│   ├── Dockerfile                   Multi-stage build → nginx
│   └── nginx.conf                   SPA + SSE-safe API proxy
├── docker-compose.yml    Full stack with persistent volume
├── vercel.json           Frontend deploy + API rewrite
├── render.yaml           Backend blueprint (Render)
├── dev.sh                Local dev shortcut
└── README.md             ← you are here
```

---

<p align="center">
  Built with 🧠 by <a href="https://github.com/officialarghya29">officialarghya29</a><br/>
  <sub>Unfold v3.1 — 68 tests, zero dependencies on LangChain</sub>
</p>
