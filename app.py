"""
MechBot - Mechanical Engineering & CNC Chatbot
==============================================

A single chat interface powered by Google Gemini. The sidebar "mode" only
changes the bot's behaviour (its system prompt):

    Tutor          -> explains mechanical engineering concepts
    CNC Assistant  -> explains / writes / debugs G-code (with a toolpath plot)
    Viva           -> acts as a viva examiner, one question at a time

Run with:   streamlit run app.py
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import streamlit as st

from utils.gcode_plot import extract_gcode_blocks, format_stats, parse_gcode, plot_toolpath
from utils.llm import MODEL_NAME, friendly_error, get_client, stream_reply
from utils.prompts import MODE_ICONS, MODES, STARTERS, build_system_prompt
from utils.rag import NotesIndex, extract_text

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="MechBot", page_icon="⚙️", layout="centered")

AVATARS = {"user": "🧑‍🎓", "assistant": "⚙️"}

st.markdown(
    """
    <style>
    .mode-badge {
        display: inline-block; padding: 0.2rem 0.75rem; border-radius: 999px;
        background: #0b5394; color: #ffffff; font-size: 0.85rem; font-weight: 600;
    }
    .mode-note {
        text-align: center; color: #5d6d7e; font-size: 0.8rem; margin: 0.4rem 0;
    }
    .footer { text-align: center; color: #7f8c8d; font-size: 0.8rem; margin-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state (the chatbot's memory between reruns)
# ---------------------------------------------------------------------------
# messages: list of {"role": "user" | "assistant" | "note", "content": str,
#                    "sources": [optional list of retrieved note chunks]}

defaults = {
    "messages": [],
    "mode": MODES[0],
    "pending_prompt": None,   # set when a starter button is clicked
    "notes_index": None,      # NotesIndex built from the uploaded file
    "notes_signature": None,  # (file name, size) so we only re-index on a new upload
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_api_key() -> tuple[str, str]:
    """Read the Gemini key from Streamlit secrets (or an env variable as a fallback).

    Returns (key, problem). If no usable key is found, key is "" and problem
    explains why, so the user knows what to fix. Secret VALUES are never shown.
    """
    key, problem = "", ""
    try:
        secrets = dict(st.secrets)
        key = str(secrets.get("GEMINI_API_KEY", ""))
        if not key:
            # Also accept the key placed under a [section] by mistake.
            for value in secrets.values():
                if hasattr(value, "get") and value.get("GEMINI_API_KEY"):
                    key = str(value["GEMINI_API_KEY"])
                    break
        if not key:
            names = ", ".join(f"`{n}`" for n in secrets) or "none"
            problem = f"Secrets loaded, but no `GEMINI_API_KEY` entry was found (secret names found: {names})."
    except Exception as exc:  # no secrets file, or invalid TOML
        problem = f"Secrets could not be loaded ({type(exc).__name__}). Check the TOML format: `GEMINI_API_KEY = \"...\"` with straight double quotes."

    key = (key or os.environ.get("GEMINI_API_KEY", "")).strip()
    if key.lower().startswith("your-"):  # placeholder from the example file
        return "", "`GEMINI_API_KEY` still contains the placeholder text — replace it with your real key."
    return key, ("" if key else problem)


@st.cache_resource
def cached_client(api_key: str):
    """Create the Gemini client once and reuse it across reruns."""
    return get_client(api_key)


def on_mode_change():
    """Called when the sidebar radio changes: leave a small note in the chat."""
    mode = st.session_state.mode
    st.session_state.messages.append(
        {"role": "note", "content": f"{MODE_ICONS[mode]} Switched to {mode} mode"}
    )


def queue_prompt(text: str):
    """Starter buttons put their question here; it is sent on this rerun."""
    st.session_state.pending_prompt = text


def clear_chat():
    st.session_state.messages = []


def build_transcript(messages: list[dict], markdown: bool) -> str:
    """Turn the chat history into a downloadable .md or .txt transcript."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# MechBot conversation ({stamp})" if markdown else f"MechBot conversation ({stamp})", ""]
    for msg in messages:
        if msg["role"] == "note":
            lines.append(f"_[{msg['content']}]_" if markdown else f"[{msg['content']}]")
        else:
            who = "You" if msg["role"] == "user" else "MechBot"
            lines.append(f"**{who}:**" if markdown else f"{who}:")
            lines.append(msg["content"])
        lines.append("")
    return "\n".join(lines)


def render_gcode_preview(reply: str):
    """If the reply contains ```gcode blocks, parse and plot each one."""
    try:
        blocks = extract_gcode_blocks(reply)
        for n, block in enumerate(blocks, start=1):
            result = parse_gcode(block)
            label = "📐 Toolpath preview" + (f" #{n}" if len(blocks) > 1 else "")
            with st.expander(label, expanded=True):
                if result.segments:
                    fig = plot_toolpath(result)
                    st.pyplot(fig)
                    plt.close(fig)  # free memory - Streamlit already has the image
                    st.markdown(format_stats(result))
                else:
                    st.info("G-code found, but it has no X/Y/Z moves to plot.")
                if result.warnings:
                    st.warning("Parser notes:\n" + "\n".join(f"- {w}" for w in result.warnings[:10]))
    except Exception as exc:  # the preview must never break the chat
        st.warning(f"Could not draw the toolpath preview ({exc}).")


def render_sources(sources: list[dict]):
    if not sources:
        return
    with st.expander(f"📄 Sources used ({len(sources)})"):
        for i, src in enumerate(sources, start=1):
            st.markdown(f"**{i}. {src['source']}** — relevance {src['score']:.2f}")
            st.caption(src["text"][:600] + ("…" if len(src["text"]) > 600 else ""))


def render_message(msg: dict):
    """Draw one message from the history."""
    if msg["role"] == "note":
        st.markdown(f"<div class='mode-note'>— {msg['content']} —</div>", unsafe_allow_html=True)
        return
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_gcode_preview(msg["content"])
            render_sources(msg.get("sources", []))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ MechBot")

    st.subheader("Mode")
    st.radio(
        "Choose how MechBot behaves",
        MODES,
        key="mode",
        format_func=lambda m: f"{MODE_ICONS[m]} {m}",
        on_change=on_mode_change,
    )

    st.subheader("Try asking")
    for question in STARTERS[st.session_state.mode]:
        st.button(question, key=f"starter_{question}", on_click=queue_prompt,
                  args=(question,), width="stretch")

    st.subheader("Your notes (optional)")
    uploaded = st.file_uploader("Upload PDF or TXT notes", type=["pdf", "txt"])
    if uploaded is None:
        st.session_state.notes_index = None
        st.session_state.notes_signature = None
    else:
        signature = (uploaded.name, uploaded.size)
        if signature != st.session_state.notes_signature:
            try:
                text = extract_text(uploaded.name, uploaded.getvalue())
                st.session_state.notes_index = NotesIndex(text, uploaded.name)
                st.session_state.notes_signature = signature
            except Exception as exc:
                st.session_state.notes_index = None
                st.error(f"Could not read that file: {exc}")
        index = st.session_state.notes_index
        if index is not None:
            if index.chunks:
                st.success(f"Indexed {len(index.chunks)} chunks from {index.source_name}")
            else:
                st.warning("No text found in that file (scanned PDFs are not supported).")

    st.subheader("Chat")
    st.button("🗑️ Clear chat", on_click=clear_chat, width="stretch")
    col_md, col_txt = st.columns(2)
    col_md.download_button("⬇️ .md", build_transcript(st.session_state.messages, markdown=True),
                           file_name="mechbot_chat.md", mime="text/markdown", width="stretch")
    col_txt.download_button("⬇️ .txt", build_transcript(st.session_state.messages, markdown=False),
                            file_name="mechbot_chat.txt", mime="text/plain", width="stretch")

    with st.expander("ℹ️ About"):
        st.markdown(
            f"""
**MechBot** is an AI chatbot for Mechanical Engineering students.

- **Tutor** – concepts with intuition, formulas and worked examples
- **CNC Assistant** – G/M codes, program debugging, toolpath preview
- **Viva** – mock oral exam with scores out of 10

Model: `{MODEL_NAME}` (Google Gemini)
Generated G-code must be verified in a simulator before machining.
            """
        )
    st.caption("Built by Afras | VIT Chennai")


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("⚙️ MechBot")
st.caption("Your Mechanical Engineering & CNC study buddy — ask, code, and prepare for your viva.")
mode = st.session_state.mode
st.markdown(f"<span class='mode-badge'>{MODE_ICONS[mode]} {mode} mode</span>", unsafe_allow_html=True)

if not st.session_state.messages:
    st.info("👋 Hi! Pick a mode in the sidebar, click a starter question, or type below.")

# 1) Re-draw the whole conversation (Streamlit reruns the script on every interaction).
for message in st.session_state.messages:
    render_message(message)

# 2) Get new input: typed text, or a starter button that was just clicked.
prompt = st.chat_input(f"Ask MechBot ({mode} mode)…")
if not prompt and st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

# 3) Answer it.
if prompt:
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    api_key, key_problem = get_api_key()
    if not api_key:
        st.error(
            "🔑 No Gemini API key found. Create `.streamlit/secrets.toml` with "
            "`GEMINI_API_KEY = \"...\"` (see `secrets.toml.example`), or add it under "
            "**Secrets** on Streamlit Community Cloud, then reload the page.\n\n"
            f"**Details:** {key_problem}"
        )
    else:
        # Optional RAG: find the most relevant chunks of the uploaded notes.
        sources = []
        index = st.session_state.notes_index
        if index is not None:
            sources = [
                {"source": index.source_name, "score": score, "text": chunk}
                for chunk, score in index.retrieve(prompt, k=3)
            ]
        system_prompt = build_system_prompt(mode, [s["text"] for s in sources])

        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            pieces = []  # collect streamed text so we keep it even if the stream fails midway

            def collect_stream():
                client = cached_client(api_key)
                for piece in stream_reply(client, system_prompt, st.session_state.messages):
                    pieces.append(piece)
                    yield piece

            try:
                with st.spinner("MechBot is thinking…"):
                    st.write_stream(collect_stream())
            except Exception as exc:
                st.error(friendly_error(exc))

            reply = "".join(pieces)
            if reply:
                render_gcode_preview(reply)
                render_sources(sources)
                st.session_state.messages.append(
                    {"role": "assistant", "content": reply, "sources": sources}
                )

st.markdown("<div class='footer'>Built by Afras | VIT Chennai</div>", unsafe_allow_html=True)
