"""
Streamlit chat interface — entry point.
Run with: streamlit run app.py
"""

import base64
import re
import uuid
from pathlib import Path

import streamlit as st

from agents.orchestrator import chat
from data.ingest import ingest

IMAGES_DIR = Path(__file__).parent / "images"


def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def extract_article_ids(text: str) -> list[str]:
    """Return unique article IDs (10-digit, leading 0) mentioned in text."""
    return list(dict.fromkeys(re.findall(r"\b0\d{9}\b", text)))


def render_product_images(article_ids: list[str]):
    """Display product images in a row for any IDs that have a local image file."""
    available = [aid for aid in article_ids if (IMAGES_DIR / f"{aid}.jpg").exists()]
    if not available:
        return
    cols = st.columns(min(len(available), 4))
    for col, aid in zip(cols, available):
        with col:
            st.image(str(IMAGES_DIR / f"{aid}.jpg"), caption=aid, use_container_width=True)

# ---------------------------------------------------------------------------
# One-time setup
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Checking product catalog...")
def startup():
    ingest()

startup()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Fashion Assistant",
    page_icon="👗",
    layout="centered",
)
st.title("👗 Fashion Assistant")
st.caption("Ask me about women's clothing — I'll help you find the perfect outfit.")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []  # [{role, content, image_bytes?}]

# ---------------------------------------------------------------------------
# Render conversation history
# ---------------------------------------------------------------------------

for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        if msg.get("image_bytes"):
            st.image(msg["image_bytes"], width=260)
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_product_images(extract_article_ids(msg["content"]))

# ---------------------------------------------------------------------------
# Input area
# ---------------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Attach an image (optional)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

user_input = st.chat_input("What are you looking for today?")

if user_input or uploaded_file:
    image_bytes = uploaded_file.read() if uploaded_file else None
    image_base64 = image_to_base64(image_bytes) if image_bytes else None
    text = user_input or ""

    # Show user message
    with st.chat_message("user"):
        if image_bytes:
            st.image(image_bytes, width=260)
        if text:
            st.markdown(text)

    st.session_state.display_messages.append({
        "role": "user",
        "content": text,
        "image_bytes": image_bytes,
    })

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner(""):
            response = chat(
                message=text,
                thread_id=st.session_state.thread_id,
                image_base64=image_base64,
            )
        st.markdown(response)
        render_product_images(extract_article_ids(response))

    st.session_state.display_messages.append({
        "role": "assistant",
        "content": response,
    })

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Session")
    st.caption(f"Thread: `{st.session_state.thread_id[:8]}…`")

    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.display_messages = []
        st.rerun()
