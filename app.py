"""
Streamlit interface for the Big Tech Financial Intelligence RAG.
"""

from __future__ import annotations

import streamlit as st

from src.generation.generate_answer import generate_answer


st.set_page_config(
    page_title="Big Tech Financial Intelligence",
    page_icon="📊",
    layout="wide",
)


def initialize_session() -> None:
    """Create the chat history when the app starts."""

    if "messages" not in st.session_state:
        st.session_state.messages = []


def display_sources(sources: list[dict]) -> None:
    """Display retrieved SEC sources."""

    if not sources:
        st.warning("No supporting sources were returned.")
        return

    with st.expander("View SEC sources"):
        seen = set()

        for source in sources:
            key = (
                source["label"],
                source["source_url"],
            )

            if key in seen:
                continue

            seen.add(key)

            description = (
                f"**[{source['label']}] "
                f"{source['ticker']}**"
            )

            if source.get("section"):
                description += (
                    f" — {source['section']}"
                )

            st.markdown(description)

            st.markdown(
                f"[Open SEC filing]"
                f"({source['source_url']})"
            )


def display_message(message: dict) -> None:
    """Render one stored chat message."""

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            display_sources(message["sources"])


initialize_session()

st.title("Big Tech Financial Intelligence")

st.caption(
    "Grounded answers from Apple, Microsoft, Alphabet, "
    "Amazon, and Meta SEC filings."
)

with st.sidebar:
    st.header("Settings")

    top_k = st.slider(
        "Narrative passages",
        min_value=1,
        max_value=10,
        value=5,
        help=(
            "Number of filing passages retrieved for "
            "narrative questions."
        ),
    )

    st.markdown(
        """
        **Retrieval system**

        - Structured SEC XBRL facts for numbers
        - Semantic search for filing explanations
        - Local Qwen model through Ollama
        - Evidence labels for every material claim
        """
    )

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

for stored_message in st.session_state.messages:
    display_message(stored_message)

question = st.chat_input(
    "Ask a question about Big Tech financial filings"
)

if question:
    user_message = {
        "role": "user",
        "content": question,
    }

    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(
            "Retrieving SEC evidence and generating answer..."
        ):
            try:
                result = generate_answer(
                    question=question,
                    top_k=top_k,
                )

                st.markdown(result["answer"])

                display_sources(result["sources"])

                assistant_message = {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                }

                st.session_state.messages.append(
                    assistant_message
                )

            except Exception as error:
                error_message = (
                    "The system could not generate an answer. "
                    f"Details: {error}"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )