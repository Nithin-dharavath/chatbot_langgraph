"""
Streamlit frontend for the LangGraph chat bot.

Run the FastAPI backend first:
    uvicorn main:app --reload --port 8080

Then start this UI:
    streamlit run streamlit_frontend.py
"""

import streamlit as st
import requests


st.set_page_config(
    page_title="LangGraph Chat Bot",
    page_icon="💬",
    layout="centered",
)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    backend_url = st.text_input(
        "Backend URL",
        value="http://localhost:8080",
        help="URL where the FastAPI backend is running.",
    )

    st.divider()
    st.caption("Start the backend with:\n\n`uvicorn main:app --reload --port 8080`")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------- State ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- Helpers ----------------
def stream_chat(message: str, url: str):
    """Yield chunks of the assistant reply coming from the backend."""
    endpoint = url.rstrip("/") + "/chat"
    try:
        with requests.post(
            endpoint,
            json={"human_message": message},
            stream=True,
            timeout=120,
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    yield chunk
    except requests.exceptions.ConnectionError:
        yield "\n\n⚠️ Could not connect to the backend. Is it running?"
    except requests.exceptions.Timeout:
        yield "\n\n⚠️ Request timed out. Please try again."
    except requests.exceptions.HTTPError as exc:
        yield f"\n\n⚠️ Server error: {exc}"
    except Exception as exc:  # noqa: BLE001
        yield f"\n\n⚠️ Unexpected error: {exc}"


# ---------------- Chat UI ----------------
st.title("💬 LangGraph Chat Bot")
st.caption("A simple chat interface powered by a LangGraph + Groq backend.")

# Render existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new input
if user_input := st.chat_input("Type your message and press Enter..."):
    # Echo the user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream the assistant reply
    with st.chat_message("assistant"):
        reply = st.write_stream(stream_chat(user_input, backend_url))

    if reply:
        st.session_state.messages.append({"role": "assistant", "content": reply})