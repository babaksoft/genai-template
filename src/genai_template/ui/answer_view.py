"""Streamlit rendering helpers for citation-aware answers."""

import streamlit as st

from genai_template.schemas import AnswerResponse


def render_answer(answer_result: AnswerResponse) -> None:
    """Render an answer, its citation warnings, and retrieved sources.

    Args:
        answer_result:
            Citation-aware response returned by the answer API.
    """

    st.subheader("Answer")
    st.markdown(answer_result.answer)

    for warning in answer_result.citation_warnings:
        labels = ", ".join(warning.labels)
        st.warning(f"{warning.message} Unsupported labels: {labels}")

    st.subheader("Sources")
    if not answer_result.sources:
        st.info("No sources were retrieved for this answer.")
        return

    for source in answer_result.sources:
        status = "Cited" if source.cited else "Uncited"
        section = f" · {source.section}" if source.section else ""
        title = (
            f"[{source.label}] {status} · {source.document_name}{section} "
            f"· distance {source.distance:.4f}"
        )
        with st.expander(title):
            st.text(source.content)
