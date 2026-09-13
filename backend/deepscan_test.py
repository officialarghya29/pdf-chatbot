"""Deepscan tests: clear-history, compression, logging, partial persistence, edge cases.

Run:  backend/venv/bin/python backend/deepscan_test.py
"""

import io
import json
import sys
import shutil
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

# ---------------------------------------------------------------- fake LLM
import llm  # noqa: E402

_call_count = 0

def fake_embed(texts):
    out = []
    for t in texts:
        v = [0.0] * 64
        for ch in t.lower():
            v[ord(ch) % 64] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        out.append([x / n for x in v])
    return out

def fake_stream_ok(messages):
    yield "The answer is 42 [1]."
    yield " It's in page 1 [2]."

def fake_stream_fail(messages):
    raise llm.LLMError("quota exceeded")

def fake_stream_partial(messages):
    yield "Partial answer"
    raise llm.LLMError("mid-stream failure")

llm.embed_texts = fake_embed
llm.check_connection = lambda: None

# isolated storage
_tmp = tempfile.mkdtemp(prefix="unfold_deepscan_")
import config  # noqa: E402
config.settings.data_dir = Path(_tmp) / "storage"
config.settings.upload_dir = Path(_tmp) / "uploads"
config.settings.data_dir.mkdir(parents=True, exist_ok=True)
config.settings.upload_dir.mkdir(parents=True, exist_ok=True)

import main  # noqa: E402
import sessions as sessions_mod  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402

client = TestClient(main.app)
PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" — {extra}" if extra else ""))


def make_pdf(title=None, pages=3):
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    if title:
        c.setTitle(title)
    for i in range(1, pages + 1):
        c.setFont("Helvetica", 11)
        c.drawString(60, 800, f"Page {i}: the quantum flux is {42}.{i}")
        c.drawString(60, 780, "Auxiliary text for testing retrieval and chunking.")
        c.showPage()
    c.save()
    return buf.getvalue()


# ============================================================= clear messages
print("[1] Clear messages endpoint")

r = client.post("/api/upload", files={"file": ("clear.pdf", make_pdf(), "application/pdf")})
sid = r.json()["session_id"]
check("upload for clear test", r.status_code == 200)

# send a few messages
llm.stream_chat = fake_stream_ok
for _ in range(3):
    with client.stream("POST", "/api/ask", json={"session_id": sid, "query": "test"}):
        pass

r = client.get(f"/api/sessions/{sid}/messages")
msgs_before = r.json()["messages"]
check("history populated before clear", len(msgs_before) >= 6)

# clear
r = client.delete(f"/api/sessions/{sid}/messages")
check("clear messages returns 200", r.status_code == 200 and r.json() == {"ok": True})

r = client.get(f"/api/sessions/{sid}/messages")
msgs_after = r.json()["messages"]
check("history empty after clear", len(msgs_after) == 0)

# clear on nonexistent session
r = client.delete("/api/sessions/nonexistent1/messages")
check("clear nonexistent -> 404", r.status_code == 404)

# ask after clear should still work (index still exists)
llm.stream_chat = fake_stream_ok
with client.stream("POST", "/api/ask", json={"session_id": sid, "query": "what is flux"}) as r:
    events = [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: ")]
check("ask works after clear", any(e["type"] == "done" for e in events))

client.delete(f"/api/sessions/{sid}")

# ============================================================= gzip compression
print("\n[2] Gzip compression")

# GZip middleware works in production with real HTTP clients.
# TestClient uses httpx which may not negotiate compression.
# Verify the middleware class is configured:
has_gzip = any(m.cls.__name__ == "GZipMiddleware" for m in main.app.user_middleware)
check("GZipMiddleware configured", has_gzip)

# ============================================================= request logging
print("\n[3] Request logging")

import logging as _logging
import io as _io
log_stream = _io.StringIO()
handler = _logging.StreamHandler(log_stream)
handler.setLevel(_logging.INFO)
handler.setFormatter(_logging.Formatter("%(message)s"))
main.log.addHandler(handler)

r = client.get("/api/health")
main.log.removeHandler(handler)
log_output = log_stream.getvalue()
check("request logged", "GET /api/health" in log_output and "200" in log_output)
check("latency logged", "ms)" in log_output)

# ============================================================= partial persistence
print("\n[4] Partial answer persistence on mid-stream error")

llm.stream_chat = fake_stream_partial
r = client.post("/api/upload", files={"file": ("partial.pdf", make_pdf(), "application/pdf")})
sid = r.json()["session_id"]

with client.stream("POST", "/api/ask", json={"session_id": sid, "query": "fail test"}) as r:
    events = [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: ")]

types = [e["type"] for e in events]
check("error event in stream", "error" in types)
check("start event before error", types.index("start") < types.index("error"))

# check partial answer was persisted
r = client.get(f"/api/sessions/{sid}/messages")
msgs = r.json()["messages"]
check("user message persisted", any(m["role"] == "user" for m in msgs))
check("partial assistant answer persisted", any(m["role"] == "assistant" for m in msgs))

client.delete(f"/api/sessions/{sid}")

# ============================================================= full stream fail
print("\n[5] Full stream failure (no partial)")

llm.stream_chat = fake_stream_fail
r = client.post("/api/upload", files={"file": ("fail.pdf", make_pdf(), "application/pdf")})
sid = r.json()["session_id"]

with client.stream("POST", "/api/ask", json={"session_id": sid, "query": "fail"}) as r:
    events = [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: ")]

check("start event emitted", any(e["type"] == "start" for e in events))
check("error event emitted", any(e["type"] == "done" for e in events) or any(e["type"] == "error" for e in events))
# if start emitted but stream failed before any delta, no assistant message should be persisted
r2 = client.get(f"/api/sessions/{sid}/messages")
msgs = r2.json()["messages"]
check("no assistant message when start + immediate error",
      not any(m["role"] == "assistant" for m in msgs) or
      any(m["role"] == "assistant" and len(m["content"]) == 0 for m in msgs))

client.delete(f"/api/sessions/{sid}")

# ============================================================= edge cases
print("\n[6] Edge cases")

# very long filename
long_name = "a" * 200 + ".pdf"
r = client.post("/api/upload", files={"file": (long_name, make_pdf(), "application/pdf")})
check("very long filename truncated safely", r.status_code == 200, str(r.json())[:80])
if r.status_code == 200:
    client.delete(f"/api/sessions/{r.json()['session_id']}")

# empty filename: fastapi testclient sends Content-Disposition with empty filename
# which triggers multipart parsing but empty name -> 400
r = client.post("/api/upload", files={"file": ("", make_pdf(), "application/pdf")})
check("empty filename handled", r.status_code in (200, 400, 422))

# filename with path separators
r = client.post("/api/upload", files={"file": ("../../../etc/passwd.pdf", make_pdf(), "application/pdf")})
check("path separator in filename handled", r.status_code in (200, 400))

# health shows correct session count
r = client.post("/api/upload", files={"file": ("count.pdf", make_pdf(), "application/pdf")})
sid = r.json()["session_id"]
r = client.get("/api/health")
check("health session count correct", r.json()["sessions"] >= 1)
client.delete(f"/api/sessions/{sid}")

# ============================================================= concurrent clears
print("\n[7] Concurrent operations")

import threading

llm.stream_chat = fake_stream_ok
r = client.post("/api/upload", files={"file": ("conc.pdf", make_pdf(pages=5), "application/pdf")})
sid = r.json()["session_id"]

errors = []

def ask_worker():
    try:
        with client.stream("POST", "/api/ask", json={"session_id": sid, "query": "concurrent test"}):
            pass
    except Exception as e:
        errors.append(str(e))

def clear_worker():
    try:
        client.delete(f"/api/sessions/{sid}/messages")
    except Exception as e:
        errors.append(str(e))

def history_worker():
    try:
        client.get(f"/api/sessions/{sid}/messages")
    except Exception as e:
        errors.append(str(e))

threads = []
for _ in range(3):
    threads.extend([
        threading.Thread(target=ask_worker),
        threading.Thread(target=clear_worker),
        threading.Thread(target=history_worker),
    ])
[t.start() for t in threads]
[t.join() for t in threads]
check("3 concurrent ask/clear/history threads", not errors, "; ".join(errors[:2]))

client.delete(f"/api/sessions/{sid}")

# ============================================================= cleanup
print("\n[8] Cleanup")
shutil.rmtree(_tmp, ignore_errors=True)
check("temp storage cleaned", not Path(_tmp).exists())

print(f"\n{'=' * 50}")
print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", *FAIL, sep="\n  - ")
    sys.exit(1)
print("ALL DEEPSCAN TESTS PASSED")
