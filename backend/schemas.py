"""Pydantic models shared across the API."""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------- requests
class AskRequest(BaseModel):
    session_id: str = Field(..., min_length=6, max_length=64)
    query: str = Field(..., min_length=1, max_length=4000)

    @field_validator("query")
    @classmethod
    def _strip_and_require(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query must not be blank.")
        return v


# ---------------------------------------------------------------- responses
class Source(BaseModel):
    page: int
    snippet: str


class Citation(BaseModel):
    n: int
    page: int


class AskResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[Citation] = []
    sources: list[Source] = []
    doc_title: Optional[str] = None


class SessionSummary(BaseModel):
    session_id: str
    title: str
    pages: int
    chunks: int
    created_at: float
    message_count: int


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_configured: bool
    sessions: int


class ErrorResponse(BaseModel):
    detail: str
