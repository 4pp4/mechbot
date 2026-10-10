"""
Thin wrapper around the Google Gemini API (official `google-genai` SDK).

Keeping all API code here means app.py never talks to Gemini directly,
and the model name lives in exactly one place.
"""

from __future__ import annotations

import time
from typing import Iterator

import httpx
from google import genai
from google.genai import errors, types

# Change the model here and nowhere else.
MODEL_NAME = "gemini-3.6-flash"

# Give up on a request after this many milliseconds instead of hanging forever.
REQUEST_TIMEOUT_MS = 60_000

# How many times to retry when Gemini says it is overloaded (HTTP 503).
MAX_RETRIES = 3


def get_client(api_key: str) -> genai.Client:
    """Create a Gemini client from an API key."""
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))


def to_gemini_history(messages: list[dict]) -> list[types.Content]:
    """Convert our chat history into Gemini's format.

    Our messages look like {"role": "user" | "assistant" | "note", "content": "..."}.
    Gemini uses the roles "user" and "model". Mode-switch notes are UI-only,
    so they are not sent to the model.
    """
    contents = []
    for msg in messages:
        if msg["role"] not in ("user", "assistant"):
            continue
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
    return contents


def stream_reply(client: genai.Client, system_prompt: str, messages: list[dict]) -> Iterator[str]:
    """Send the whole conversation to Gemini and yield the reply piece by piece.

    If Gemini is overloaded (HTTP 503) before any text arrives, wait and retry
    a few times. Once text has been shown we never retry, to avoid duplicates.
    """
    config = types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.4)
    contents = to_gemini_history(messages)
    for attempt in range(MAX_RETRIES + 1):
        started = False
        try:
            stream = client.models.generate_content_stream(model=MODEL_NAME, contents=contents, config=config)
            for chunk in stream:
                if chunk.text:  # some chunks (e.g. safety metadata) carry no text
                    started = True
                    yield chunk.text
            return
        except errors.ServerError:
            if started or attempt == MAX_RETRIES:
                raise
            time.sleep(2 * (attempt + 1))  # 2 s, 4 s, 6 s ...


def friendly_error(exc: Exception) -> str:
    """Turn any exception into a short, student-friendly message."""
    if isinstance(exc, errors.ClientError):
        code = getattr(exc, "code", None)
        text = str(exc).lower()
        reason = getattr(exc, "message", "") or ""  # Google's own explanation
        if code == 429 or "resource_exhausted" in text:
            return "⏳ Gemini rate limit reached. Please wait a minute and try again (the free tier allows a limited number of requests per minute)."
        if code == 404:
            return f"❓ The model '{MODEL_NAME}' is not available. Update MODEL_NAME in utils/llm.py.\n\nGoogle says: {reason}"
        if code in (401, 403) or "api key not valid" in text or "api_key_invalid" in text:
            return f"🔑 Your Gemini API key was rejected. Check GEMINI_API_KEY in .streamlit/secrets.toml (or in the Streamlit Cloud Secrets).\n\nGoogle says: {reason}"
        return f"⚠️ Gemini rejected the request ({code}). Try rephrasing or clearing the chat.\n\nGoogle says: {reason}"
    if isinstance(exc, errors.ServerError):
        return "🛠️ Gemini is overloaded right now (Google's side, not your code). Please try again in a minute."
    if isinstance(exc, httpx.TimeoutException):
        return "⌛ Gemini took too long to answer. Please try again (or try a shorter question)."
    if isinstance(exc, (httpx.ConnectError, httpx.NetworkError)):
        return "🌐 Network error - could not reach Gemini. Check your internet connection and try again."
    return f"⚠️ Something went wrong: {type(exc).__name__}: {exc}"
