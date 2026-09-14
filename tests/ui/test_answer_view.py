"""Tests for citation-aware Streamlit answer rendering."""

from unittest.mock import MagicMock, call, patch

from genai_template.schemas import (
    AnswerResponse,
    CitationSource,
    CitationWarning,
    RunMetrics,
)
from genai_template.ui.answer_view import render_answer


def create_response(
    *,
    sources: list[CitationSource] | None = None,
    warnings: list[CitationWarning] | None = None,
) -> AnswerResponse:
    """Create an answer response for presentation tests.

    Args:
        sources:
            Ordered citation sources to render.
        warnings:
            Citation warnings to render.

    Returns:
        Complete citation-aware answer response.
    """

    return AnswerResponse(
        answer="A **Markdown** answer [S1].",
        metrics=RunMetrics(
            query="Question",
            embedding_model="embedder",
            vector_store="Chroma",
            llm_model="llm",
            top_k=5,
            retrieved_chunks=len(sources or []),
            context_length=0,
            prompt_length=10,
            response_length=6,
            retrieval_time=0.1,
            generation_time=0.2,
            total_time=0.3,
        ),
        sources=sources or [],
        citation_warnings=warnings or [],
    )


@patch("genai_template.ui.answer_view.st")
def test_render_answer_displays_markdown_and_ordered_sources(st: MagicMock) -> None:
    """The answer and all sources should retain API ordering and exact content."""

    sources = [
        CitationSource(
            label="S1",
            chunk_id="guide-001",
            document_name="guide.md",
            section="/Introduction/",
            content="First exact chunk.",
            distance=0.12,
            cited=True,
        ),
        CitationSource(
            label="S2",
            chunk_id="guide-002",
            document_name="guide.md",
            section=None,
            content="Second exact chunk.",
            distance=0.4,
            cited=False,
        ),
    ]

    render_answer(create_response(sources=sources))

    st.markdown.assert_called_once_with("A **Markdown** answer [S1].")
    assert st.expander.call_args_list == [
        call("[S1] Cited · guide.md · /Introduction/ · distance 0.1200"),
        call("[S2] Uncited · guide.md · distance 0.4000"),
    ]
    assert st.text.call_args_list == [
        call("First exact chunk."),
        call("Second exact chunk."),
    ]


@patch("genai_template.ui.answer_view.st")
def test_render_answer_displays_warnings_and_empty_source_state(
    st: MagicMock,
) -> None:
    """Citation warnings and empty retrievals should remain visible."""

    warning = CitationWarning(
        code="unsupported_citation_labels",
        message="The answer references labels not present in its context.",
        labels=["S9", "S10"],
    )

    render_answer(create_response(warnings=[warning]))

    st.warning.assert_called_once_with(
        "The answer references labels not present in its context. "
        "Unsupported labels: S9, S10"
    )
    st.info.assert_called_once_with("No sources were retrieved for this answer.")
    st.expander.assert_not_called()
