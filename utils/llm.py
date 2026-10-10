"""
Thin wrapper around the Google Gemini API (official `google-genai` SDK).

Keeping all API code here means app.py never talks to Gemini directly,
and the model name lives in exactly one place.
"""

from __future__ import annotations

from typing import Iterator

import httpx
from google import genai
from google.genai import errors, types

# Change the model here and nowhere else.
MODEL_NAME = "gemini-2.5-flash"


def get_client(api_key: str) -> genai.Client:
    """Create a Gemini client from an API key."""
    return genai.Client(api_key=api_key)


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
    """Send the whole conversation to Gemini and yield the reply piece by piece."""
    config = types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.4)
    stream = client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=to_gemini_history(messages),
        config=config,
    )
    for chunk in stream:
        if chunk.text:  # some chunks (e.g. safety metadata) carry no text
            yield chunk.text


def friendly_error(exc: Exception) -> str:
    """Turn any exception into a short, student-friendly message."""
    if isinstance(exc, errors.ClientError):
        code = getattr(exc, "code", None)
        text = str(exc).lower()
        if code == 429 or "resource_exhausted" in text:
            return "⏳ Gemini rate limit reached. Please wait a minute and try again (the free tier allows a limited number of requests per minute)."
        if code in (401, 403) or "api key" in text or "api_key" in text:
            return "🔑 Your Gemini API key was rejected. Check GEMINI_API_KEY in .streamlit/secrets.toml (or in the Streamlit Cloud Secrets)."
        if code == 404:
            return f"❓ The model '{MODEL_NAME}' was not found. Update MODEL_NAME in utils/llm.py."
        return f"⚠️ Gemini rejected the request ({code}). Try rephrasing or clearing the chat."
    if isinstance(exc, errors.ServerError):
        return "🛠️ Gemini's servers are busy right now. Please try again in a few seconds."
    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError)):
        return "🌐 Network error - could not reach Gemini. Check your internet connection and try again."
    return f"⚠️ Something went wrong: {type(exc).__name__}: {exc}"
