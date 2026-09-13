"""Session management, FAISS vector search, chat history and persistence.

One `Session` per uploaded document. Everything is persisted to disk so a
server restart does not lose your chat.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import faiss

import llm
from config import settings
from ingest import Chunk

log = logging.getLogger("neochat.sessions")

MAX_SESSIONS = 30

_PROMPT_HEADER = "You are NeoChat, an AI assistant that answers questions about a specific PDF document."


class Session:
    """A single document chat with vector index and message history."""

    def __init__(self, session_id: str, title: str, filename: str, pages: int):
        self.id = session_id
        self.title = title
        self.filename = filename
        self.pages = pages
        self.created_at = time.time()
        self.messages: list[dict] = []
        self._chunks: list[Chunk] = []
        self._index: Optional[faiss.Index] = None
        self._embeddings: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._dirty = False

    # ------------------------------------------------------------- index
    def build_index(self, chunks: list[Chunk]) -> None:
        texts = [c.text for c in chunks]
        vectors = np.asarray(llm.embed_texts(texts), dtype="float32")
        faiss.normalize_L2(vectors)

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        self._chunks = chunks
        self._index = index
        self._embeddings = vectors
        self._dirty = True

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    # ------------------------------------------------------------- search
    def search(self, query: str, k: int | None = None) -> list[dict]:
        if self._index is None:
            return []
        k = k or settings.top_k
        qvec = np.asarray(llm.embed_texts([query]), dtype="float32")
        faiss.normalize_L2(qvec)
        scores, ids = self._index.search(qvec, min(k, max(1, len(self._chunks))))
        hits: list[dict] = []
        for rank, (score, idx) in enumerate(zip(scores[0], ids[0])):
            if idx == -1:
                continue
            c = self._chunks[idx]
            hits.append({"rank": rank, "score": float(score), "text": c.text, "page": c.page})
        return hits

    # ------------------------------------------------------------- history
    def add_message(self, role: str, content: str) -> None:
        with self._lock:
            self.messages.append({"role": role, "content": content})
            self._dirty = True
        self._persist()

    # ------------------------------------------------------------- prompt
    def build_messages(self, query: str) -> tuple[list[dict], list[dict]]:
        """Returns (messages_for_llm, sources_used)."""
        hits = self.search(query)
        history = self.messages[-settings.max_history * 2 :]

        context_parts = [
            f"[{i+1}] (page {h['page']})\n{h['text']}" for i, h in enumerate(hits)
        ]
        context = "\n\n".join(context_parts) if context_parts else "(no context found)"

        system = (
            f"{_PROMPT_HEADER}\n"
            "Answer ONLY using the provided CONTEXT from the document.\n"
            "Cite sources inline like [1], [2] matching the numbered context blocks.\n"
            "If the context does not contain the answer, say you could not find it "
            "in the document and briefly suggest what the user could ask instead.\n"
            "Be concise, accurate, and use markdown when helpful.\n\n"
            f"DOCUMENT: \"{self.title}\" ({self.pages} pages)\n\n"
            f"CONTEXT:\n{context}"
        )

        messages = [{"role": "system", "content": system}]
        for m in history:
            if m["role"] in ("user", "assistant"):
                messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": query})
        return messages, hits

    # ------------------------------------------------------------- persistence
    def _persist(self) -> None:
        try:
            dir_ = settings.data_dir / self.id
            dir_.mkdir(parents=True, exist_ok=True)
            payload = {
                "id": self.id,
                "title": self.title,
                "filename": self.filename,
                "pages": self.pages,
                "created_at": self.created_at,
                "messages": self.messages,
                "chunks": [
                    {"text": c.text, "page": c.page, "index": c.index}
                    for c in self._chunks
                ],
                "embeddings_file": "embeddings.npy" if self._embeddings is not None else None,
            }
            (dir_ / "meta.json").write_text(json.dumps(payload))
            if self._embeddings is not None:
                np.save(dir_ / "embeddings.npy", self._embeddings)
            self._dirty = False
        except Exception as exc:  # persistence must never crash the app
            log.warning("Failed to persist session %s: %s", self.id, exc)

    @classmethod
    def load(cls, session_id: str) -> Optional["Session"]:
        d = settings.data_dir / session_id
        meta_path = d / "meta.json"
        if not meta_path.exists():
            return None
        try:
            data = json.loads(meta_path.read_text())
            s = cls(data["id"], data["title"], data["filename"], data["pages"])
            s.created_at = data.get("created_at", s.created_at)
            s.messages = data.get("messages", [])
            chunks = [
                Chunk(text=c["text"], page=c["page"], index=c["index"])
                for c in data.get("chunks", [])
            ]
            emb_file = d / "embeddings.npy"
            if chunks and emb_file.exists():
                vectors = np.load(emb_file)
                index = faiss.IndexFlatIP(vectors.shape[1])
                index.add(vectors)
                s._chunks = chunks
                s._embeddings = vectors
                s._index = index
            return s
        except Exception as exc:
            log.warning("Failed to load session %s: %s", session_id, exc)
            return None

    def delete_disk(self) -> None:
        shutil.rmtree(settings.data_dir / self.id, ignore_errors=True)


class SessionManager:
    """Thread-safe in-memory registry backed by disk."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def all(self) -> list[Session]:
        with self._lock:
            return sorted(self._sessions.values(), key=lambda s: s.created_at, reverse=True)

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            s = self._sessions.get(session_id)
        if s is None:
            # try loading from disk (e.g. after a server restart)
            s = Session.load(session_id)
            if s is not None:
                with self._lock:
                    self._sessions[session_id] = s
        return s

    def add(self, session: Session) -> None:
        with self._lock:
            self._sessions[session.id] = session
        self._enforce_limit()

    def remove(self, session_id: str) -> bool:
        with self._lock:
            s = self._sessions.pop(session_id, None)
        if s:
            s.delete_disk()
            return True
        # maybe exists only on disk
        if (settings.data_dir / session_id).exists():
            shutil.rmtree(settings.data_dir / session_id, ignore_errors=True)
            return True
        return False

    def _enforce_limit(self) -> None:
        with self._lock:
            if len(self._sessions) <= MAX_SESSIONS:
                return
            oldest = sorted(self._sessions.values(), key=lambda s: s.created_at)[0]
            del self._sessions[oldest.id]
        oldest.delete_disk()
        log.info("Evicted oldest session %s (limit %d)", oldest.id, MAX_SESSIONS)


manager = SessionManager()
