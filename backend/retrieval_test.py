"""Retrieval engine tests: hybrid search, BM25, MMR, and Markdown export.

Fully offline — embeddings are stubbed with controlled vectors so ranking is
deterministic and the LLM is never contacted.

Run:  backend/venv/bin/python backend/retrieval_test.py
"""

from contextlib import contextmanager
from pathlib import Path
import sys

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

import numpy as np  # noqa: E402
import faiss  # noqa: E402

import llm  # noqa: E402
from config import settings  # noqa: E402
from ingest import Chunk  # noqa: E402
from sessions import Session, manager, _LexicalStats, _minmax, _tokenize  # noqa: E402

# --------------------------------------------------------------- fake embed
# A single mutable vector every query embeds to, so each test controls the
# "dense" signal exactly.
QUERY_VEC = [1.0, 0.0, 0.0]


def fake_embed(texts):
    return [list(QUERY_VEC) for _ in texts]


llm.embed_texts = fake_embed

PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'✅' if cond else '❌'} {name}" + (f" — {extra}" if extra else ""))


@contextmanager
def override(**kw):
    """Temporarily override settings attributes."""
    saved = {k: getattr(settings, k) for k in kw}
    for k, value in kw.items():
        setattr(settings, k, value)
    try:
        yield
    finally:
        for k, value in saved.items():
            setattr(settings, k, value)


def build_session(chunks, vectors, session_id="retrieval-test", title="Doc"):
    """A Session with hand-set chunks/embeddings, bypassing LLM indexing."""
    s = Session(session_id, title, "doc.pdf", 1)
    v = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(v)
    index = faiss.IndexFlatIP(v.shape[1])
    index.add(v)
    s._chunks = chunks
    s._embeddings = v
    s._index = index
    s._lex_stats = None
    return s


def chunk(text, page):
    return Chunk(text=text, page=page, index=0)


# ------------------------------------------------------------- pure helpers
print("\n== helpers ==")
check("_minmax normalises a range", _minmax([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0])
check("_minmax on a flat list yields zeros", _minmax([3.0, 3.0, 3.0]) == [0.0, 0.0, 0.0])
check("_minmax on empty input yields empty", _minmax([]) == [])
check("_tokenize drops single characters", _tokenize("a bb ccc 42") == ["bb", "ccc", "42"])
check("_tokenize lowercases", _tokenize("Mixed CASE") == ["mixed", "case"])

stats = _LexicalStats([chunk("mitochondria powerhouse", 1), chunk("plain filler words", 2)])
check("BM25 scores one value per chunk", len(stats.scores("mitochondria")) == 2)
check("BM25 ignores unknown terms", stats.scores("zzzz") == [0.0, 0.0])
check("BM25 scores nothing for an empty query", stats.scores("") == [0.0, 0.0])
rare = stats.scores("mitochondria")
check("BM25 favours the matching chunk", rare[0] > rare[1], str(rare))

# ---------------------------------------------------- lexical beats dense
print("\n== hybrid retrieval ==")
chunks = [
    chunk("alpha beta gamma delta epsilon", 1),
    chunk("the mitochondria is the powerhouse of the cell", 2),
    chunk("unrelated filler content about nothing", 3),
]
# Dense vectors deliberately point AWAY from the correct answer for chunk 1.
vectors = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
QUERY_VEC = [1.0, 0.0, 0.0]

s = build_session(chunks, vectors)
with override(hybrid_search=False, mmr_lambda=1.0):
    dense_only = s.search("mitochondria", k=1)
check(
    "dense-only search misses the rare term",
    dense_only[0]["page"] != 2,
    f"got page {dense_only[0]['page']}",
)

with override(hybrid_search=True, hybrid_alpha=1.0, mmr_lambda=1.0):
    lexical = s.search("mitochondria", k=1)
check(
    "hybrid search recovers the rare term",
    lexical[0]["page"] == 2,
    f"got page {lexical[0]['page']}",
)

# pure dense still works when hybrid is off
QUERY_VEC = [0.0, 0.0, 1.0]
with override(hybrid_search=False, mmr_lambda=1.0):
    by_vector = s.search("anything at all", k=1)
check("dense-only ranks by embedding similarity", by_vector[0]["page"] == 2)

# ----------------------------------------------------------- MMR diversity
print("\n== MMR diversification ==")
# Chunks 10 and 11 have *identical* vectors, so their relevance scores are
# bit-for-bit equal and clearly above chunk 12. Pure relevance therefore
# returns the duplicate pair; diversification must swap one of them out.
QUERY_VEC = [3.0, 1.0, 0.0]
dup_chunks = [
    chunk("duplicate passage number one", 10),
    chunk("duplicate passage number two", 11),
    chunk("a completely different topic entirely", 12),
]
dup_vectors = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
dup_session = build_session(dup_chunks, dup_vectors, session_id="mmr-test")

with override(hybrid_search=False, mmr_lambda=1.0):
    relevance_order = dup_session.search("q", k=2)
check(
    "pure relevance picks the two duplicates",
    {h["page"] for h in relevance_order} == {10, 11},
    str([h["page"] for h in relevance_order]),
)

with override(hybrid_search=False, mmr_lambda=0.5):
    diverse = dup_session.search("q", k=2)
pages = [h["page"] for h in diverse]
kept_duplicates = {p for p in pages if p in (10, 11)}
check("MMR drops a duplicate for a distinct passage", 12 in pages, str(pages))
check(
    "MMR still keeps one of the most relevant passages",
    len(kept_duplicates) == 1 and len(pages) == 2,
    str(pages),
)
check("MMR never returns the duplicate pair together", kept_duplicates != {10, 11}, str(pages))

# ---------------------------------------------------------------- edges
print("\n== edge cases ==")
empty = Session("empty-test", "Empty", "e.pdf", 0)
check("search on an unindexed session returns []", empty.search("anything") == [])

with override(hybrid_search=True, mmr_lambda=0.7):
    big_k = dup_session.search("duplicate", k=99)
check("k larger than the corpus is clamped", len(big_k) == 3, str(len(big_k)))

flat = build_session(
    [chunk("same text everywhere", 1), chunk("same text everywhere", 2)],
    [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    session_id="flat-test",
)
with override(hybrid_search=True, hybrid_alpha=0.5, mmr_lambda=0.7):
    flat_hits = flat.search("same text", k=2)
check("degenerate embeddings do not crash", len(flat_hits) == 2)
check(
    "scores stay finite",
    all(np.isfinite(h["score"]) for h in flat_hits),
    str([h["score"] for h in flat_hits]),
)

with override(hybrid_search=True, hybrid_alpha=0.35, mmr_lambda=0.7):
    hits = dup_session.search("duplicate topic", k=3)
ids = [h["rank"] for h in hits]
check("ranks are 0..n-1 without duplicates", ids == list(range(len(hits))), str(ids))
check("pages are ints", all(isinstance(h["page"], int) for h in hits))

# build_messages still wires context and citations correctly
with override(hybrid_search=True, hybrid_alpha=0.35, mmr_lambda=0.7):
    messages, used = dup_session.build_messages("duplicate topic")
check("build_messages returns hits", bool(used))
check("system prompt carries numbered context", "[1] (page " in messages[0]["content"])
check("last message is the user query", messages[-1]["role"] == "user")

# ------------------------------------------------ Markdown export endpoint
print("\n== export endpoint ==")
from fastapi.testclient import TestClient  # noqa: E402
import main  # noqa: E402

client = TestClient(main.app)

export_session = Session("export-test-01", "My Research Paper", "paper.pdf", 7)
export_session.messages = [
    {"role": "user", "content": "What is the finding?"},
    {"role": "assistant", "content": "It is 12% [1]."},
]
manager.add(export_session)
try:
    r = client.get("/api/sessions/export-test-01/export")
    check("GET export = 200", r.status_code == 200, str(r.status_code))
    check("export is markdown", "text/markdown" in r.headers.get("content-type", ""))
    check(
        "export sets a download filename",
        "attachment" in r.headers.get("content-disposition", "")
        and ".md" in r.headers.get("content-disposition", ""),
        r.headers.get("content-disposition", ""),
    )
    body = r.text
    check("export includes the title", "# My Research Paper" in body)
    check("export includes document metadata", "paper.pdf" in body)
    check("export marks both speakers", "### You" in body and "### Unfold" in body)
    check("export preserves the answer", "It is 12% [1]." in body)
finally:
    manager.remove(export_session.id)

check("export on a missing session = 404", client.get("/api/sessions/nope/export").status_code == 404)

print("\n" + "=" * 48)
print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:")
    for name in FAIL:
        print(f"  - {name}")
    sys.exit(1)
print("ALL RETRIEVAL TESTS PASSED")
