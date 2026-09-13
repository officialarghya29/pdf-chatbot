"""PDF ingestion: text extraction, cleaning, chunking and embeddings."""

import re
from dataclasses import dataclass, field

from pypdf import PdfReader

from config import settings


@dataclass
class Chunk:
    text: str
    page: int          # 1-based page number
    index: int         # position within the document


@dataclass
class IngestResult:
    filename: str
    title: str
    pages: int
    chunks: list[Chunk] = field(default_factory=list)


_HEADER_FOOTER = re.compile(
    r"^(?:page\s*\d+(?:\s*/\s*\d+)?|\d+\s*|\d+\s*/\s*\d+)$", re.IGNORECASE
)


def _clean_line(line: str) -> bool:
    """Drop page numbers / running headers & footers."""
    s = line.strip()
    if not s:
        return False
    if _HEADER_FOOTER.match(s):
        return False
    return True


def _clean_text(raw: str) -> str:
    lines = [ln.rstrip() for ln in raw.splitlines()]
    kept = [ln for ln in lines if _clean_line(ln)]
    text = "\n".join(kept)
    # collapse 3+ blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # soft-fix hyphenated line breaks: "docu-\nment" -> "document"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return text.strip()


def _fallback_title(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    stem = stem.replace("_", " ").replace("-", " ").strip()
    return stem[:80] or "Untitled document"


def extract_pdf(path: str, original_filename: str) -> IngestResult:
    reader = PdfReader(path)
    if reader.is_encrypted:
        try:
            reader.decrypt("")
            reader.decrypt("")
        except Exception as exc:  # pragma: no cover - pypdf raises various types
            raise ValueError(
                "This PDF is password protected. Please upload an unlocked copy."
            ) from exc

    n_pages = len(reader.pages)
    if n_pages == 0:
        raise ValueError("The PDF contains no pages.")

    # ---- metadata title
    title = ""
    try:
        meta_title = (reader.metadata or {}).get("/Title", "") or ""
        if meta_title.strip():
            title = meta_title.strip()[:120]
    except Exception:
        pass
    if not title:
        title = _fallback_title(original_filename)

    # ---- page-wise extraction
    page_texts: list[str] = []
    total_chars = 0
    for page in reader.pages:
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        cleaned = _clean_text(raw)
        page_texts.append(cleaned)
        total_chars += len(cleaned)

    if total_chars < 25:
        raise ValueError(
            "No extractable text found (this PDF is likely a scan). "
            "Please upload a text-based PDF."
        )

    # ---- chunking: page-aware sliding window on words
    chunks: list[Chunk] = []
    approx_words = max(1, settings.chunk_size // 6)  # ~6 chars/word
    overlap_words = max(0, settings.chunk_overlap // 6)

    for pno, ptext in enumerate(page_texts, start=1):
        if not ptext:
            continue
        words = ptext.split()
        if not words:
            continue

        step = max(1, approx_words - overlap_words)
        local_chunks: list[tuple[int, str]] = []
        i = 0
        while i < len(words):
            window = words[i : i + approx_words]
            text = " ".join(window)
            if len(text) < 25:
                break
            local_chunks.append((i, text))
            if i + approx_words >= len(words):
                break
            i += step

        for pos, text in local_chunks:
            chunks.append(Chunk(text=text, page=pno, index=len(chunks)))

    if not chunks:
        raise ValueError("Could not build any text chunks from this PDF.")

    return IngestResult(
        filename=original_filename,
        title=title,
        pages=n_pages,
        chunks=chunks,
    )
