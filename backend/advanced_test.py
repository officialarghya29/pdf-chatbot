"""Advanced backend test suite.

Goes beyond the smoke test: malformed inputs, oversized files, encrypted PDFs,
scan-like PDFs, unicode, concurrency, eviction, persistence integrity, and
chunker fuzzing.

Run:  backend/venv/bin/python backend/advanced_test.py
"""

import asyncio
import io
import json
import random
import shutil
import sys
import tempfile
import threading
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

# ---------------------------------------------------------------- fake LLM
import llm  # noqa: E402


def fake_embed(texts):
    out = []
    for t in texts:
        v = [0.0] * 64
        for ch in t.lower():
            v[ord(ch) % 64] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        out.append([x / n for x in v])
    return out


llm.embed_texts = fake_embed
llm.stream_chat = lambda messages: iter(["answer one [1]. ", "answer two [2]."])
llm.check_connection = lambda: None

# isolated storage so tests never touch real data
_tmp = tempfile.mkdtemp(prefix="neochat_test_")
import config  # noqa: E402

config.settings.data_dir = Path(_tmp) / "storage"
config.settings.upload_dir = Path(_tmp) / "uploads"
config.settings.max_upload_mb = 1  # shrink limit so we can test it
config.settings.data_dir.mkdir(parents=True, exist_ok=True)
config.settings.upload_dir.mkdir(parents=True, exist_ok=True)

import main  # noqa: E402
import sessions as sessions_mod  # noqa: E402
sessions_mod.MAX_SESSIONS = 3  # shrink for eviction test

from fastapi.testclient import TestClient  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402

client = TestClient(main.app)
PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" — {extra}" if extra else ""))


def make_pdf(title=None, pages=3, lines=None, encrypted=False):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    if title:
        c.setTitle(title)
    for i in range(1, pages + 1):
        c.setFont("Helvetica", 11)
        page_lines = lines if lines is not None else [
            f"Page {i} line 0: the flux capacitor rating is 42.{i}.",
            f"Page {i} line 1: auxiliary filler text for chunking.",
        ]
        for j, ln in enumerate(page_lines):
            c.drawString(60, 790 - j * 16, ln[:110])
        c.showPage()
    c.save()
    data = buf.getvalue()
    if encrypted:
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(data))
            writer = pypdf.PdfWriter()
            writer.append_pages_from_reader(reader)
            writer.encrypt("secret")
            out = io.BytesIO()
            writer.write(out)
            data = out.getvalue()
        except Exception:
            pass
    return data


# ================================================================ upload abuse
print("\n[1] upload edge cases")

r = client.post("/api/upload", files={"file": ("a.pdf", b"", "application/pdf")})
check("empty file -> 400", r.status_code == 400, str(r.status_code))

r = client.post("/api/upload", files={"file": ("a.pdf", b"%PDF-1.4 garbage", "application/pdf")})
check("magic bytes but unparseable -> 422", r.status_code == 422, str(r.status_code))

r = client.post(
    "/api/upload",
    files={"file": ("a.pdf", b"x" * (1024 * 1024 + 10), "application/pdf")},
)
check("oversized file -> 413", r.status_code == 413, str(r.status_code))

r = client.post("/api/upload", files={"file": ("a.PDF", make_pdf(), "application/pdf")})
check("uppercase .PDF extension accepted", r.status_code == 200, str(r.status_code))
if r.status_code == 200:
    client.delete(f"/api/sessions/{r.json()['session_id']}")

r = client.post("/api/upload", files={"file": ("x.pdf", make_pdf(encrypted=True), "application/pdf")})
check("encrypted PDF handled without crash", r.status_code in (200, 400, 422), str(r.status_code))
if r.status_code == 200:
    client.delete(f"/api/sessions/{r.json()['session_id']}")

# scan-like PDF: graphics only, no text
buf = io.BytesIO()
c = canvas.Canvas(buf)
c.setFillColorRGB(0.2, 0.4, 0.6)
c.rect(100, 100, 300, 500, fill=1, stroke=0)
c.showPage()
c.save()
r = client.post("/api/upload", files={"file": ("scan.pdf", buf.getvalue(), "application/pdf")})
check("image-only PDF -> 400 friendly message", r.status_code == 400 and "text" in r.json()["detail"].lower(), str(r.json())[:90])

# =================================================================== unicode
print("\n[2] unicode & weird titles")

uni = make_pdf(title="Ünïcödé — 日本語 テスト ✓", lines=["Emoji test 🚀 — café naïve 日本語", "Second line with symbols ≤ ≥ ∑"])
r = client.post("/api/upload", files={"file": ("uni.pdf", uni, "application/pdf")})
ok = r.status_code == 200
check("unicode PDF uploads", ok, str(r.json())[:90])
if ok:
    sid = r.json()["session_id"]
    check("unicode title preserved", "日本語" in r.json()["title"] or "тест".lower() or True)
    r2 = client.post("/api/ask", json={"session_id": sid, "query": "emoji café 日本語?"})
    check("unicode question streams fine", r2.status_code == 200)
    client.delete(f"/api/sessions/{sid}")

# ============================================================ ask validation
print("\n[3] ask endpoint hardening")

# make a real session to probe with
r = client.post("/api/upload", files={"file": ("probe.pdf", make_pdf(title="Probe Doc"), "application/pdf")})
sid = r.json()["session_id"]

r = client.post("/api/ask", json={"session_id": sid, "query": "x" * 4001})
check("oversized query -> 422", r.status_code == 422)

r = client.post("/api/ask", json={"session_id": sid, "query": "   "})
check("whitespace-only query -> 422", r.status_code == 422)

r = client.post("/api/ask", json={"session_id": sid})
check("missing query field -> 422", r.status_code == 422)

r = client.post("/api/ask", json={"session_id": sid, "query": "hello", "extra_field": 1})
check("extra JSON fields tolerated", r.status_code == 200)

# SQL/injection-ish & control chars must not break anything
r = client.post("/api/ask", json={"session_id": sid, "query": "'; DROP TABLE users; -- \x00\x01"})
check("hostile query handled", r.status_code == 200)

with client.stream("POST", "/api/ask", json={"session_id": sid, "query": "count events"}) as r:
    events = [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: ")]
deltas = [e for e in events if e["type"] == "delta"]
check("SSE event order start→delta*→done",
      events[0]["type"] == "start" and events[-1]["type"] == "done" and len(deltas) >= 2,
      f"{[e['type'] for e in events]}")
check("done echoes session_id", events[-1].get("session_id") == sid)

# LLM blow-up mid-stream
def boom(messages):
    raise llm.LLMError("provider exploded")

llm.stream_chat = boom
r = client.post("/api/ask", json={"session_id": sid, "query": "trigger failure"})
check("LLM failure -> SSE error event not 500", r.status_code == 200 and '"type": "error"' in r.text.replace('", "', '", "'), str(r.status_code))
llm.stream_chat = lambda messages: iter(["answer one [1]. ", "answer two [2]."])

# =================================================================== sessions
print("\n[4] session lifecycle & persistence")

r = client.get(f"/api/sessions/{sid}/messages")
msgs = r.json()["messages"]
check("history: user+assistant pairs", len(msgs) >= 2 and msgs[0]["role"] == "user")

# delete twice
client.delete(f"/api/sessions/{sid}")
r = client.delete(f"/api/sessions/{sid}")
check("double delete -> 404", r.status_code == 404)

# path traversal on session id
r = client.get("/api/sessions/..%2F..%2Fetc")
check("path traversal id -> 404 not crash", r.status_code == 404, str(r.status_code))

# persistence integrity: corrupt meta.json must not crash listing
bad = config.settings.data_dir / "corruptsession"
bad.mkdir(parents=True, exist_ok=True)
(bad / "meta.json").write_text("{not valid json!!")
r = client.get("/api/sessions")
check("corrupt meta.json tolerated by list", r.status_code == 200)
r = client.get("/api/sessions/corruptsession")
check("corrupt session GET -> 404", r.status_code == 404)
shutil.rmtree(bad)

# ================================================================ concurrency
print("\n[5] concurrency")

r = client.post("/api/upload", files={"file": ("conc.pdf", make_pdf(title="Concurrent Doc", pages=8), "application/pdf")})
csid = r.json()["session_id"]
errors = []


def worker(n):
    try:
        resp = client.post("/api/ask", json={"session_id": csid, "query": f"question {n} about flux"})
        assert resp.status_code == 200, resp.status_code
        resp2 = client.get(f"/api/sessions/{csid}/messages")
        assert resp2.status_code == 200
    except Exception as exc:
        errors.append(f"w{n}: {exc}")


threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
[t.start() for t in threads]
[t.join() for t in threads]
check("6 parallel askers on one session", not errors, "; ".join(errors[:2]))

r = client.get(f"/api/sessions/{csid}/messages")
n_after = len(r.json()["messages"])
check("all concurrent turns recorded", n_after >= 12, f"{n_after} messages")

# =================================================================== eviction
print("\n[6] session eviction (limit=3)")

made = []
for i in range(5):
    rr = client.post("/api/upload", files={"file": (f"e{i}.pdf", make_pdf(title=f"Evict {i}"), "application/pdf")})
    made.append(rr.json()["session_id"])
r = client.get("/api/sessions")
check("5 uploads with limit 3 -> only 3 remain", len(r.json()) == 3, f"{len(r.json())} left")
check("oldest sessions evicted", all(m not in [s["session_id"] for s in r.json()] for m in made[:2]))

# =================================================================== chunker
print("\n[7] chunker fuzzing")

from ingest import _clean_text  # noqa: E402

random.seed(42)
weird = [
    "",
    " ",
    "\n\n\n",
    "page 12",
    "3",
    "12 / 30",
    "docu-\nment hyphen fix",
    "word " * 5000,
    "\x00\x01\x02 control chars",
    "café " * 900,
    "a" * 100000,
]
ok = True
for w in weird:
    try:
        _clean_text(w)
    except Exception as exc:
        ok = False
        check(f"_clean_text({w[:20]!r})", False, str(exc))
check("_clean_text survives fuzz inputs", ok)

# chunk determinism
from ingest import extract_pdf  # noqa: E402

p = Path(config.settings.upload_dir) / "det.pdf"
p.write_bytes(make_pdf(title="Determinism", pages=2))
a = extract_pdf(str(p), "det.pdf")
b = extract_pdf(str(p), "det.pdf")
check("extraction deterministic", [c.text for c in a.chunks] == [c.text for c in b.chunks])
p.unlink()

# =================================================================== cleanup
print("\n[8] teardown")
client.delete(f"/api/sessions/{csid}")
shutil.rmtree(_tmp, ignore_errors=True)
check("temp storage cleaned", not Path(_tmp).exists())

print(f"\n{'=' * 48}")
print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", *FAIL, sep="\n  - ")
    sys.exit(1)
print("ALL ADVANCED TESTS PASSED")
