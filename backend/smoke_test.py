"""End-to-end smoke test: real PDF parsing + full API flow with a mocked LLM.

Run:  backend/venv/bin/python backend/smoke_test.py
"""

import asyncio
import io
import json
import sys
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


def fake_stream(messages):
    yield "Quantum flux "
    yield "is 42.0 [1]."


llm.embed_texts = fake_embed
llm.stream_chat = fake_stream
llm.chat = lambda messages, stream=False: "Quantum flux is 42.0 [1]."
llm.check_connection = lambda: None

# ------------------------------------------------------------- build test PDF
from reportlab.pdfgen import canvas  # noqa: E402

buf = io.BytesIO()
c = canvas.Canvas(buf)
c.setTitle("Quantum Flux Field Manual")
for i in range(1, 6):
    c.setFont("Helvetica", 12)
    c.drawString(72, 800, f"Quantum Flux Manual — page {i}")
    c.drawString(72, 770, f"The quantum flux constant on page {i} is 42.{i}.")
    c.drawString(72, 750, "Auxiliary background text used to pad the chunk.")
    c.showPage()
c.save()
PDF_BYTES = buf.getvalue()
Path("/tmp/smoke.pdf").write_bytes(PDF_BYTES)

# ------------------------------------------------------------------- run API
from fastapi.testclient import TestClient  # noqa: E402
import main  # noqa: E402

client = TestClient(main.app)
PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'✅' if cond else '❌'} {name}" + (f" — {extra}" if extra else ""))


print("\n== health ==")
r = client.get("/api/health")
check("GET /api/health = 200", r.status_code == 200, str(r.json()))

print("\n== upload ==")
r = client.post(
    "/api/upload",
    files={"file": ("flux_manual.pdf", PDF_BYTES, "application/pdf")},
)
check("POST /api/upload = 200", r.status_code == 200, str(r.json())[:120])
s = r.json()
sid = s["session_id"]
check("title from PDF metadata", s["title"] == "Quantum Flux Field Manual", s["title"])
check("5 pages parsed", s["pages"] == 5, f"pages={s['pages']}")
check("chunks > 0", s["chunks"] > 0, f"chunks={s['chunks']}")

print("\n== upload rejections ==")
r = client.post("/api/upload", files={"file": ("x.txt", b"hello", "text/plain")})
check("non-PDF rejected (400)", r.status_code == 400, str(r.json())[:80])
r = client.post("/api/upload", files={"file": ("fake.pdf", b"not a pdf at all", "application/pdf")})
check("corrupt PDF rejected (400)", r.status_code == 400, str(r.json())[:80])

print("\n== ask (streaming) ==")
with client.stream("POST", "/api/ask", json={"session_id": sid, "query": "What is the quantum flux?"}) as r:
    check("POST /api/ask = 200", r.status_code == 200)
    events = []
    for line in r.iter_lines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
types = [e["type"] for e in events]
check("SSE has start event", "start" in types, str(types))
check("SSE has delta events", "delta" in types)
check("SSE has done event", "done" in types)
done = next((e for e in events if e["type"] == "done"), {})
check("citations extracted [1]->page", done.get("citations"), str(done.get("citations")))
start = next((e for e in events if e["type"] == "start"), {})
check("sources attached to start", bool(start.get("sources")))

print("\n== session management ==")
r = client.get("/api/sessions")
check("list sessions contains new one", any(x["session_id"] == sid for x in r.json()))
r = client.get(f"/api/sessions/{sid}/messages")
check("history has 2 messages", len(r.json()["messages"]) == 2)
r = client.post("/api/ask", json={"session_id": "nonexistent1", "query": "hi"})
check("unknown session = 404", r.status_code == 404)
r = client.post("/api/ask", json={"session_id": sid, "query": ""})
check("empty query = 422", r.status_code == 422)

print("\n== persistence round-trip ==")
main.manager._sessions.clear()
r = client.get(f"/api/sessions/{sid}")
check("session reloaded from disk", r.status_code == 200, str(r.json())[:100])

print("\n== delete ==")
r = client.delete(f"/api/sessions/{sid}")
check("delete = 200", r.status_code == 200)
r = client.get(f"/api/sessions/{sid}")
check("deleted session = 404", r.status_code == 404)

print(f"\n{'='*44}\nRESULT: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL SMOKE TESTS PASSED ✨")
