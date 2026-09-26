"""Session management, FAISS vector search, chat history and persistence.

One `Session` per uploaded document. Everything is persisted to disk so a
server restart does not lose your chat.
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import faiss

import llm
from config import settings
from ingest import Chunk

log = logging.getLogger("unfold.sessions")

MAX_SESSIONS = 30

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, discarding single characters."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1]


def _minmax(values: list[float]) -> list[float]:
    """Scale to [0, 1]; a flat input maps to all zeros (never divides by zero)."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


class _LexicalStats:
    """Cached BM25 statistics over one session's chunks.

    BM25 complements embeddings: it rewards exact term matches and is not
    fooled by a query whose meaning is close but whose wording is rare.
    """

    K1 = 1.5
    B = 0.75

    def __init__(self, chunks: list[Chunk]):
        self.size = len(chunks)
        self.term_freqs: list[Counter] = []
        self.doc_freq: Counter = Counter()
        self.lengths: list[int] = []
        for chunk in chunks:
            tokens = _tokenize(chunk.text)
            freq = Counter(tokens)
            self.term_freqs.append(freq)
            self.doc_freq.update(freq.keys())
            self.lengths.append(len(tokens))
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0

    def scores(self, query: str) -> list[float]:
        """One BM25 score per chunk, in chunk order."""
        n = self.size
        if n == 0:
            return []
        terms = set(_tokenize(query))
        out = [0.0] * n
        if not terms or self.avg_len <= 0:
            return out

        for term in terms:
            df = self.doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for i, freq in enumerate(self.term_freqs):
                f = freq.get(term, 0)
                if not f:
                    continue
                norm = 1.0 - self.B + self.B * (self.lengths[i] / self.avg_len)
                out[i] += idf * (f * (self.K1 + 1.0)) / (f + self.K1 * norm)
        return out


def _mmr_select(
    candidate_ids: list[int],
    relevance: dict[int, float],
    embeddings: "np.ndarray",
    k: int,
    lam: float,
) -> list[int]:
    """Greedy Maximal Marginal Relevance selection.

    Picks the highest-scoring candidate, then repeatedly picks the candidate
    that best balances relevance against similarity to what is already
    chosen, so the retrieved passages are not near-duplicates of each other.
    """
    selected: list[int] = []
    remaining = list(candidate_ids)
    while remaining and len(selected) < k:
        best_id = remaining[0]
        best_value: Optional[float] = None
        for cid in remaining:
            if selected:
                redundancy = max(float(embeddings[cid] @ embeddings[s]) for s in selected)
            else:
                redundancy = 0.0
            value = lam * relevance[cid] - (1.0 - lam) * redundancy
            if best_value is None or value > best_value:
                best_value = value
                best_id = cid
        selected.append(best_id)
        remaining.remove(best_id)
    return selected


_PROMPT_HEADER = "You are Unfold, an expert AI assistant that answers questions about a specific PDF document."


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
        self._lex_stats: Optional[_LexicalStats] = None
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
        self._lex_stats = None  # rebuilt lazily on the next search
        self._dirty = True

    def _lexical(self) -> _LexicalStats:
        """BM25 statistics for the current chunks, built once and cached."""
        if self._lex_stats is None or self._lex_stats.size != len(self._chunks):
            self._lex_stats = _LexicalStats(self._chunks)
        return self._lex_stats

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    # ------------------------------------------------------------- search
    def search(self, query: str, k: int | None = None) -> list[dict]:
        """Retrieve the most useful passages for a query.

        Three stages: a dense candidate pool from FAISS, an optional BM25
        blend, then MMR diversification. Falls back to plain vector search
        when hybrid retrieval is disabled and MMR is off.
        """
        if self._index is None or not self._chunks:
            return []

        n = len(self._chunks)
        k = max(1, min(k or settings.top_k, n))

        qvec = np.asarray(llm.embed_texts([query]), dtype="float32")
        faiss.normalize_L2(qvec)

        # Candidate pool: generous enough that a strong lexical match which
        # dense search ranks low still gets considered.
        pool = min(n, max(k * 8, 40))
        raw_scores, raw_ids = self._index.search(qvec, pool)
        cand_ids = [int(i) for i in raw_ids[0] if i != -1]
        dense = {int(i): float(s) for s, i in zip(raw_scores[0], raw_ids[0]) if i != -1}
        if not cand_ids:
            return []

        if settings.hybrid_search:
            lexical = self._lexical().scores(query)
            dense_norm = _minmax([dense[i] for i in cand_ids])
            lex_norm = _minmax([lexical[i] for i in cand_ids])
            alpha = min(max(settings.hybrid_alpha, 0.0), 1.0)
            combined = {
                cid: (1.0 - alpha) * d + alpha * l
                for cid, d, l in zip(cand_ids, dense_norm, lex_norm)
            }
        else:
            combined = dict(dense)

        if settings.mmr_lambda < 1.0 and self._embeddings is not None and len(cand_ids) > k:
            order = _mmr_select(
                cand_ids, combined, self._embeddings, k, max(0.0, settings.mmr_lambda)
            )
        else:
            order = sorted(cand_ids, key=lambda i: combined[i], reverse=True)[:k]

        return [
            {
                "rank": rank,
                "score": float(combined[idx]),
                "text": self._chunks[idx].text,
                "page": self._chunks[idx].page,
            }
            for rank, idx in enumerate(order)
        ]

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
                s._lex_stats = None
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
