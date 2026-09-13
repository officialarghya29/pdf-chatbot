"""OpenAI-compatible LLM client: chat, embeddings, and token counting."""

from __future__ import annotations

import tiktoken
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError, RateLimitError, AuthenticationError

from config import settings


class LLMError(Exception):
    """Raised for any failure talking to the LLM provider."""


def _client() -> OpenAI:
    if not settings.openai_api_key:
        raise LLMError(
            "OPENAI_API_KEY is not configured. Add it to backend/.env and restart."
        )
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=settings.request_timeout,
        max_retries=2,
    )


def check_connection() -> None:
    """Cheap readiness probe."""
    _client()


def chat(messages: list[dict], stream: bool = False):
    """Send a chat completion request. Returns content str, or an iterator when streaming."""
    try:
        resp = _client().chat.completions.create(
            model=settings.chat_model,
            messages=messages,
            temperature=settings.temperature,
            stream=stream,
        )
    except (AuthenticationError, RateLimitError, APITimeoutError,
            APIConnectionError, APIError) as exc:
        raise LLMError(_friendly(exc)) from exc
    except Exception as exc:  # network misconfig etc.
        raise LLMError(f"LLM request failed: {exc}") from exc

    if stream:
        return resp
    return resp.choices[0].message.content or ""


def stream_chat(messages: list[dict]):
    """Yield incremental text deltas from a streaming chat completion."""
    stream = chat(messages, stream=True)
    for event in stream:
        if event.choices:
            delta = event.choices[0].delta
            if delta and delta.content:
                yield delta.content


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed texts; returns one vector per input."""
    if not texts:
        return []
    out: list[list[float]] = []
    B = 64
    for i in range(0, len(texts), B):
        batch = texts[i : i + B]
        try:
            resp = _client().embeddings.create(
                model=settings.embedding_model, input=batch
            )
        except (AuthenticationError, RateLimitError, APITimeoutError,
                APIConnectionError, APIError) as exc:
            raise LLMError(_friendly(exc)) from exc
        except Exception as exc:
            raise LLMError(f"Embedding request failed: {exc}") from exc
        out.extend(d.embedding for d in resp.data)
    return out


_ENCODER = None


def token_len(text: str) -> int:
    global _ENCODER
    if _ENCODER is None:
        try:
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODER = None
            return max(1, len(text) // 4)
    if _ENCODER is None:
        return max(1, len(text) // 4)
    return len(_ENCODER.encode(text))


def _friendly(exc: Exception) -> str:
    if isinstance(exc, AuthenticationError):
        return "Invalid or missing OPENAI_API_KEY. Check backend/.env."
    if isinstance(exc, RateLimitError):
        return "LLM provider rate limit / quota exceeded. Please retry shortly."
    if isinstance(exc, APITimeoutError):
        return "The LLM provider took too long to respond. Please retry."
    if isinstance(exc, APIConnectionError):
        return "Could not reach the LLM provider. Check your network."
    return f"LLM provider error: {exc}"
