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
  <strong>AI-powered PDF chat</strong> — upload any document, ask questions in natural language,<br/>
  get answers with inline citations mapped to exact pages. No LangChain, no bloat.
</p>

<p align="center">
  <a href="#what-is-unfold">What is it</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#features">Features</a> ·
  <a href="#api-reference">API</a> ·
  <a href="#deployment">Deploy</a> ·
  <a href="#why-unfold">Why Unfold?</a> ·
  <a href="#troubleshooting">Troubleshooting</a>
</p>

---

## What is Unfold?

**Unfold** is a full-stack AI application that lets you **chat with your PDF documents**. Upload a research paper, textbook, legal document, or report — then ask questions in natural language. Unfold reads the document, finds relevant passages, and generates answers with **inline citations** that map to specific pages.

### The problem it solves

Traditional document reading is slow and linear. You have to:
- Scroll through 50+ pages to find one fact
- Manually cross-reference sections
- Copy-paste quotes into notes
- Re-read entire chapters to understand context

**Unfold eliminates all of that.** Ask "What are the main conclusions?" and get a cited, structured answer in 2 seconds.

### How it's different

Unlike most AI chatbots that hallucinate answers, Unfold:
1. **Retrieves real passages** from your document using vector search
2. **Cites its sources** with inline `[1]`, `[2]` markers mapped to page numbers
3. **Streams answers** in real time as the model generates them
4. **Remembers context** — multi-turn conversations about the same document
5. **Persists everything** — sessions survive server restarts

---

## How it works

### The RAG pipeline

Unfold uses **Retrieval-Augmented Generation (RAG)** — a technique that grounds LLM responses in real document content:

```
┌──────────────────────────────────────────────────────────────────┐
│                        UPLOAD PHASE                              │
│                                                                  │
│  PDF ──► pypdf extracts text per page                           │
│       ──► Cleaning: remove headers/footers, fix hyphens         │
│       ──► Chunking: 1000-word sliding window (200 overlap)      │
│       ──► Embedding: OpenAI text-embedding-3-small → vectors    │
│       ──► Indexing: FAISS cosine similarity index               │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                        QUERY PHASE                               │
│                                                                  │
│  Question ──► Embed the query → vector                          │
│           ──► FAISS search → top-5 most similar chunks          │
│           ──► Build prompt: system + context + history + query   │
│           ──► Stream LLM response token by token                │
│           ──► Extract [n] citations from answer text            │
└──────────────────────────────────────────────────────────────────┘
```

### Data flow diagram

```
User opens app          Frontend loads            Backend
     │                      │                       │
     │  ──► GET /health ──► │ ──► health check ──► │
     │                      │ ◄── {version, key} ── │
     │  ◄── UI renders ──── │                       │
     │                      │                       │
User drops PDF         XHR upload (progress %)     Backend
     │                      │                       │
     │  ──► drag & drop ──► │ ──► POST /upload ──► │
     │                      │     multipart file    │
     │                      │     ───────────────── │  pypdf → text
     │                      │                       │  chunk → vectors
     │                      │                       │  FAISS → index
     │                      │ ◄── 200 + session ── │
     │  ◄── sidebar updates │                       │
     │                      │                       │
User asks question     SSE stream                  Backend
     │                      │                       │
     │  ──► POST /ask ──────│──────────────────────►│
     │                      │  ◄── event: start ─── │  (sources found)
     │                      │  ◄── event: delta ─── │  "The answer"
     │  ◄── tokens appear   │  ◄── event: delta ─── │  " is 42"
     │  ◄── [citation] ──── │  ◄── event: done ────│  (citations)
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- An [OpenAI API key](https://platform.openai.com/api-keys) (or any compatible provider)

### 1. Clone & install

```bash
git clone https://github.com/officialarghya29/pdf-chatbot.git
cd pdf-chatbot

# Backend
cd backend
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # ← add your OPENAI_API_KEY
cd ..

# Frontend
cd frontend
npm install
cd ..
```

### 2. Run

```bash
# Terminal 1 — Backend (port 8000)
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2 — Frontend (port 5173)
cd frontend
npm run dev
```

Open **http://localhost:5173** → upload a PDF → start asking questions.

### Or use the shortcut

```bash
cp .env.example.docker .env       # set OPENAI_API_KEY
./dev.sh                          # starts both servers
```

---

## Features

### Core capabilities

| Feature | Description |
|---------|-------------|
| 📄 **PDF Ingestion** | Extracts text page-by-page, cleans headers/footers, repairs hyphens, detects encrypted/scanned PDFs |
| ✂️ **Smart Chunking** | Sliding-window word chunking with configurable size/overlap, per-page metadata preserved |
| 🔍 **Vector Search** | FAISS cosine similarity with L2-normalized embeddings, top-k configurable |
| ⚡ **Streaming Chat** | SSE token streaming — answers appear in real time as the model generates them |
| 📌 **Inline Citations** | `[1]`, `[2]` markers in answers map to exact pages via expandable source chips |
| 💾 **Persistence** | FAISS indices, chat history, and session metadata survive server restarts (disk-backed) |
| 🔒 **Concurrency-Safe** | Thread-safe session store, heavy work offloaded to worker threads |
| 🛡️ **Error Boundary** | React crash recovery UI; every OpenAI error mapped to a friendly message |
| 📱 **Mobile Ready** | Responsive sidebar with hamburger menu, works on phones and tablets |

### UX enhancements

| Feature | Description |
|---------|-------------|
| 📋 **Copy Button** | One-click copy on every assistant message with 2-second visual feedback |
| ⬇️ **Jump to Bottom** | Floating button appears when you scroll up, auto-scrolls on new tokens |
| 🧹 **Clear History** | Eraser icon per session clears chat without deleting the document |
| ⚠️ **API Key Warning** | Dismissible amber banner when OPENAI_API_KEY is missing, with setup instructions |
| 🎨 **Glassmorphism UI** | Futuristic dark theme with ambient gradients, blur effects, and animated hero |
| ⚙️ **Gzip Compression** | Automatic response compression for all API endpoints |
| 📝 **Request Logging** | Every request logged with method, path, status code, and latency |

---

## API Reference

### Endpoints

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

```
data: {"type": "start", "sources": [{"page": 3, "snippet": "..."}]}

data: {"type": "delta",  "text": "The answer"}

data: {"type": "delta",  "text": " is 42"}

data: {"type": "done",   "answer": "The answer is 42", "citations": [{"n": 1, "page": 3}]}
```

### Example: Upload → Ask flow

```bash
# Upload
curl -X POST http://localhost:8000/api/upload -F "file=@paper.pdf"
# Returns: {"session_id": "a1b2c3d4e5f6", "title": "Paper Title", "pages": 12, "chunks": 15, ...}

# Ask
curl -N -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"session_id": "a1b2c3d4e5f6", "query": "What are the key findings?"}'
# Streams: start → deltas → done with citations

# Clear history
curl -X DELETE http://localhost:8000/api/sessions/a1b2c3d4e5f6/messages

# Delete session
curl -X DELETE http://localhost:8000/api/sessions/a1b2c3d4e5f6
```

---

## Configuration

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required.** Your secret key ([get one](https://platform.openai.com/api-keys)) |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Compatible endpoint (Azure, OpenRouter, Groq) |
| `CHAT_MODEL` | `gpt-4o-mini` | Chat completion model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `TEMPERATURE` | `0.3` | Response randomness (0 = deterministic, 1 = creative) |
| `CHUNK_SIZE` | `1000` | Approximate characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `TOP_K` | `5` | Chunks retrieved per query |
| `MAX_HISTORY` | `10` | Conversation turns sent to the LLM |
| `MAX_UPLOAD_MB` | `25` | Max PDF upload size |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |

### Using other providers

Unfold works with any OpenAI-compatible API:

```bash
# OpenRouter (access GPT-4, Claude, etc.)
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=sk-or-...
CHAT_MODEL=openai/gpt-4o-mini

# Groq (fast inference)
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_API_KEY=gsk_...
CHAT_MODEL=llama-3.1-70b-versatile

# Local (Ollama)
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
CHAT_MODEL=llama3.1
EMBEDDING_MODEL=nomic-embed-text
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Frontend (Vite + React 18 + TypeScript + Tailwind CSS)            │
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
│  │ Routes: /api/health  /api/upload  /api/ask  /api/sessions  │    │
│  └────────┬──────────────────┬───────────────────┬────────────┘    │
│           │                  │                   │                   │
│  ┌────────▼─────────┐ ┌─────▼──────────┐ ┌──────▼────────────┐   │
│  │ ingest.py         │ │ sessions.py    │ │ llm.py            │   │
│  │  pypdf → text     │ │  FAISS index   │ │  OpenAI client    │   │
│  │  cleaning         │ │  search        │ │  streaming chat   │   │
│  │  chunking         │ │  persistence   │ │  embeddings       │   │
│  └──────────────────┘ │  history       │ │  error mapping    │   │
│                        └────────────────┘ └───────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

### Tech stack

| Layer | Technology | Why |
|-------|------------|-----|
| **PDF parsing** | pypdf | Lightweight, fast, handles encrypted PDFs |
| **Chunking** | Custom (no langchain) | Full control, no abstraction debt |
| **Embeddings** | OpenAI text-embedding-3-small | High quality, low cost ($0.02/1M tokens) |
| **Vector search** | FAISS IndexFlatIP | Fast cosine similarity, runs locally, no server needed |
| **LLM** | OpenAI gpt-4o-mini | Fast, cheap ($0.15/1M input), good at RAG |
| **Backend** | FastAPI + uvicorn | Async, auto-docs, type-safe, production-ready |
| **Frontend** | React 18 + TypeScript + Tailwind | Component-based, type-safe, utility CSS |
| **Deployment** | Docker + nginx + Gunicorn | Production-grade, single command |

---

## Deployment

### Option A: Docker (simplest)

```bash
cp .env.example.docker .env       # set OPENAI_API_KEY
docker compose up --build         # → http://localhost:8080
```

### Option B: Split cloud (free tiers)

| Service | Platform | Config file |
|---------|----------|-------------|
| Backend (API) | [Render](https://render.com) | `render.yaml` |
| Frontend (UI) | [Vercel](https://vercel.com) | `vercel.json` |

### Option C: Local development

```bash
./dev.sh    # starts backend :8000 + frontend :5173
```

---

## Testing

Unfold has **95 automated tests** that run fully offline (LLM is mocked):

```bash
# Backend — 71 checks
python backend/smoke_test.py        # 20 end-to-end API flow tests
python backend/advanced_test.py     # 29 abuse, concurrency, eviction, fuzzing
python backend/deepscan_test.py     # 22 clear-history, logging, persistence, edge cases

# Frontend — 24 checks (vitest + Testing Library + jsdom)
cd frontend && npm test

# Full build
cd frontend && npm run build        # strict TypeScript + production bundle
```

### What the tests cover

| Category | Tests | What they verify |
|----------|-------|-----------------|
| Health & sessions | 8 | CRUD, listing, history |
| Upload validation | 12 | Non-PDF, corrupt, oversized, encrypted, scan-only, empty, long names |
| Chat & streaming | 15 | SSE event order, citations, LLM failure, mid-stream error, partial persistence |
| Persistence | 6 | Disk round-trip, corrupt data tolerance, eviction |
| Concurrency | 4 | Parallel asks, clear+history+ask simultaneously |
| Edge cases | 16 | Path traversal, hostile queries, whitespace, unicode, extra fields |
| Component renders | 10 | MessageBubble, Composer, Sidebar, copy button, thinking state |
| SSE parsing | 7 | Chunk boundaries, truncation, malformed events, abort |

---

## Why Unfold?

### The comparison

| Feature | Unfold | Basic Chatbots | LangChain Demos | Enterprise RAG |
|---------|--------|----------------|-----------------|----------------|
| **Setup time** | 2 min | 5 min | 30+ min | Days/weeks |
| **Dependencies** | 7 packages | Varies | 20+ | Dozens |
| **LangChain** | ❌ Zero | — | ✅ Deep coupling | ✅ Deep coupling |
| **Streaming answers** | ✅ Real-time SSE | ❌ Wait for full | ⚠️ Generator | ✅ |
| **Citations → pages** | ✅ Inline `[1]` → page | ❌ None | ⚠️ Manual | ⚠️ Extra work |
| **Multi-document** | ✅ Per-session | ❌ Single | ⚠️ Manual | ✅ |
| **Persistence** | ✅ Disk-backed | ❌ Ephemeral | ❌ Ephemeral | ✅ |
| **Error handling** | ✅ Friendly messages | ❌ Tracebacks | ⚠️ Generic | ✅ |
| **Frontend quality** | ✅ Futuristic UI | ⚠️ Bare minimal | ⚠️ Streamlit | ✅ Custom |
| **Docker deploy** | ✅ One command | ❌ Manual | ❌ Manual | ⚠️ K8s |
| **Free deployment** | ✅ Vercel + Render | — | — | 💰 Paid |
| **Open source** | ✅ Full code | — | — | — |

### The philosophy

```
Traditional approach (LangChain):
  pip install langchain openai chromadb ...    ← 20+ packages
  from langchain import ...                    ← deep coupling
  chain = RetrievalQA.from_chain_type(...)     ← opaque abstraction
  result = chain.run(query)                    ← no streaming, no citations

Unfold approach (no abstractions):
  pip install openai faiss-cpu pypdf fastapi   ← 7 packages, zero coupling
  client = OpenAI(api_key=...)                 ← direct API, transparent
  index.add(embed_texts(chunks))               ← explicit vector search
  for delta in stream_chat(messages): ...      ← real streaming, full control
```

Unfold avoids framework abstraction layers. Every line of code is readable, debuggable, and replaceable. **The result is fewer bugs, faster startup, and an app that actually works.**

---

## Troubleshooting

| Error message | Cause | Fix |
|---------------|-------|-----|
| `OPENAI_API_KEY is not configured` | No key in `backend/.env` | Create `backend/.env` with `OPENAI_API_KEY=sk-...` and restart |
| `Invalid or missing OPENAI_API_KEY` | Wrong key or expired | Check at [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `Rate limit / quota exceeded` | Too many requests | Wait or upgrade your OpenAI plan |
| `Could not reach the LLM provider` | Network issue | Check internet + `OPENAI_BASE_URL` in `.env` |
| `Only PDF files are supported` | Non-PDF uploaded | Upload a `.pdf` file |
| `No extractable text found` | Scanned/image-only PDF | Upload a text-based PDF (not a scan) |
| `This PDF is password protected` | Encrypted PDF | Remove password before uploading |
| `File too large` | PDF > 25 MB | Compress or split the PDF |
| `Session not found` | Deleted or server restarted | Upload the PDF again |
| CORS errors in browser | Backend not running | Start backend on :8000, check `CORS_ORIGINS` |
| `Backend offline` in UI | Backend not started | `cd backend && uvicorn main:app --reload --port 8000` |
| Blank white page | React crash | ErrorBoundary shows recovery UI; check console |

---

## Project structure

```
unfold/
├── backend/
│   ├── main.py              FastAPI app, routes, streaming SSE
│   ├── config.py            Env-driven settings (pydantic-settings)
│   ├── ingest.py            PDF extraction, cleaning, chunking
│   ├── llm.py               OpenAI client (chat/embed/stream)
│   ├── sessions.py          Session store, FAISS search, persistence
│   ├── schemas.py           Pydantic request/response models
│   ├── smoke_test.py        20 end-to-end API tests
│   ├── advanced_test.py     29 abuse/concurrency/edge-case tests
│   ├── deepscan_test.py     22 persistence/logging/LLM-failure tests
│   ├── requirements.txt     Python dependencies (7 packages)
│   ├── .env.example         Environment template
│   ├── Dockerfile           Production container image
│   └── venv/                Virtual environment (gitignored)
├── frontend/
│   ├── src/
│   │   ├── App.tsx                  Root state + streaming orchestration
│   │   ├── main.tsx                 Entry with ErrorBoundary
│   │   ├── index.css                Tailwind theme + glassmorphism
│   │   ├── components/
│   │   │   ├── Sidebar.tsx          Sessions + clear/delete + mobile menu
│   │   │   ├── EmptyState.tsx       Animated hero + suggestion cards
│   │   │   ├── MessageBubble.tsx    Markdown + citations + copy + thinking
│   │   │   ├── Composer.tsx         Upload + message input + drag-drop
│   │   │   ├── SourcePanel.tsx      Expandable source chunks
│   │   │   ├── StatusBar.tsx        Health + connection + version
│   │   │   ├── ApiWarning.tsx       Missing API key banner
│   │   │   └── ErrorBoundary.tsx    Crash recovery UI
│   │   ├── lib/
│   │   │   ├── api.ts              Typed API client + SSE parser
│   │   │   ├── utils.ts            formatBytes, timeAgo, uid
│   │   │   ├── api.test.ts         7 SSE parser tests
│   │   │   └── utils.test.ts       3 utility tests
│   │   └── test/
│   │       └── setup.ts            Vitest setup + jest-dom
│   ├── components/*.test.tsx        14 component render tests
│   ├── index.html                   HTML entry point
│   ├── vite.config.ts               Dev proxy + test config
│   ├── tsconfig.json                Strict TypeScript
│   ├── package.json                 Dependencies + scripts
│   ├── Dockerfile                   Multi-stage build → nginx
│   └── nginx.conf                   SPA + SSE-safe API proxy
├── docker-compose.yml       Full stack with persistent volume
├── vercel.json              Frontend deploy + API rewrite
├── render.yaml              Backend blueprint (Render)
├── .dockerignore            Build context exclusions
├── .gitignore               Git exclusions
├── dev.sh                   Local dev shortcut
├── .env.example.docker      Docker env template
└── README.md                ← you are here
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing`
3. Make your changes
4. Run tests: `python backend/smoke_test.py && cd frontend && npm test`
5. Commit: `git commit -m "Add amazing feature"`
6. Push: `git push origin feature/amazing`
7. Open a Pull Request

### Development guidelines

- **Backend**: Keep routes thin, logic in services. All new endpoints must have tests.
- **Frontend**: Use TypeScript strict mode. Components must render without crashing (ErrorBoundary catches).
- **Tests**: Every bug fix should include a regression test. Run `npm test` before committing.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with 🧠 by <a href="https://github.com/officialarghya29">officialarghya29</a><br/>
  <sub>Unfold v3.1 · 95 tests · 7 backend packages · 0 LangChain dependencies</sub>
</p>
