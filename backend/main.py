"""Unfold — AI PDF chatbot API."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse

import llm
from config import settings
from ingest import extract_pdf
from schemas import (
    AskRequest,
    Citation,
    HealthResponse,
    SessionSummary,
    Source,
)
from sessions import Session, manager

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("unfold")

app = FastAPI(title=settings.app_name, version=settings.app_version)

app.add_middleware(GZipMiddleware, minimum_size=500)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    ms = (time.monotonic() - start) * 1000
    log.info("%s %s → %d (%.0fms)", request.method, request.url.path, response.status_code, ms)
    return response


PDF_MAGIC = b"%PDF-"


# ------------------------------------------------------------------ helpers
def _session_or_404(session_id: str) -> Session:
    s = manager.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return s


def _validate_upload(file: UploadFile, data: bytes) -> None:
    name = (file.filename or "").lower()
    if not name.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {settings.max_upload_mb} MB).",
        )
    if PDF_MAGIC not in data[:1024]:
        raise HTTPException(status_code=400, detail="Corrupt or invalid PDF file.")


def _summary(s: Session) -> SessionSummary:
    return SessionSummary(
        session_id=s.id,
        title=s.title,
        pages=s.pages,
        chunks=s.chunk_count,
        created_at=s.created_at,
        message_count=len(s.messages),
    )


def _extract_citations(answer: str, hits: list[dict]) -> list[Citation]:
    """Map [n] markers in the answer to page numbers."""
    pages_by_n = {i + 1: h["page"] for i, h in enumerate(hits)}
    found: dict[int, int] = {}
    for match in re.findall(r"\[(\d{1,2})\]", answer):
        n = int(match)
        if n in pages_by_n and n not in found:
            found[n] = pages_by_n[n]
    return [Citation(n=n, page=p) for n, p in sorted(found.items())]


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


# ------------------------------------------------------------------ routes
@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        llm_configured=bool(settings.openai_api_key),
        sessions=len(manager.all()),
    )


@app.get("/api/sessions", response_model=list[SessionSummary])
def list_sessions() -> list[SessionSummary]:
    return [_summary(s) for s in manager.all()]


@app.get("/api/sessions/{session_id}", response_model=SessionSummary)
def get_session(session_id: str) -> SessionSummary:
    return _summary(_session_or_404(session_id))


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    if not manager.remove(session_id):
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"ok": True}


@app.get("/api/sessions/{session_id}/messages")
def get_messages(session_id: str) -> dict:
    s = _session_or_404(session_id)
    return {"messages": s.messages}


@app.delete("/api/sessions/{session_id}/messages")
def clear_messages(session_id: str) -> dict:
    s = _session_or_404(session_id)
    s.messages = []
    s._persist()
    return {"ok": True}


@app.post("/api/upload", response_model=SessionSummary)
async def upload_pdf(file: UploadFile = File(...)) -> SessionSummary:
    data = await file.read()
    _validate_upload(file, data)

    tmp_path = settings.upload_dir / f"tmp_{uuid.uuid4().hex}.pdf"
    tmp_path.write_bytes(data)

    try:
        result = await asyncio.to_thread(
            extract_pdf, str(tmp_path), file.filename or "document.pdf"
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("PDF parsing failed")
        raise HTTPException(status_code=422, detail="Could not parse this PDF.") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    session = Session(
        session_id=uuid.uuid4().hex[:12],
        title=result.title,
        filename=result.filename,
        pages=result.pages,
    )

    try:
        await asyncio.to_thread(session.build_index, result.chunks)
    except llm.LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    manager.add(session)
    session._persist()
    log.info("Session %s created: %s (%d chunks)", session.id, session.title, session.chunk_count)
    return _summary(session)


@app.post("/api/ask")
async def ask(req: AskRequest) -> StreamingResponse:
    s = _session_or_404(req.session_id)

    def event_stream():
        try:
            messages, hits = s.build_messages(req.query)
        except llm.LLMError as exc:
            yield _sse({"type": "error", "detail": str(exc)})
            return

        s.add_message("user", req.query)

        yield _sse({
            "type": "start",
            "sources": [
                Source(page=h["page"], snippet=h["text"][:280]).model_dump()
                for h in hits
            ],
        })

        answer_parts: list[str] = []
        persisted = False

        def persist_answer() -> None:
            nonlocal persisted
            if persisted:
                return
            persisted = True
            answer = "".join(answer_parts).strip()
            if answer:
                s.add_message("assistant", answer)

        try:
            for delta in llm.stream_chat(messages):
                answer_parts.append(delta)
                yield _sse({"type": "delta", "text": delta})
        except llm.LLMError as exc:
            yield _sse({"type": "error", "detail": str(exc)})
            persist_answer()
            return
        except (GeneratorExit, asyncio.CancelledError):
            persist_answer()
            raise

        persist_answer()

        answer = "".join(answer_parts).strip()

        yield _sse({
            "type": "done",
            "session_id": s.id,
            "answer": answer,
            "citations": [c.model_dump() for c in _extract_citations(answer, hits)],
        })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def root() -> dict:
    return {"name": settings.app_name, "version": settings.app_version, "status": "running"}
