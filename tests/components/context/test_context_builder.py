"""Tests for the ContextBuilder."""

from genai_template.components.context import ContextBuilder
from genai_template.schemas import (
    DocumentChunk,
    RetrievedChunk,
)


def create_retrieved_chunk(
    text: str,
    *,
    distance: float = 0.0,
) -> RetrievedChunk:
    """Create a RetrievedChunk for testing."""

    chunk = DocumentChunk(
        id="chunk-id",
        document_id="document-id",
        text=text,
        metadata={"source": "test.txt"},
        embedding=None,
    )

    return RetrievedChunk(
        chunk=chunk,
        distance=distance,
    )


def test_build_returns_empty_string_for_empty_input() -> None:
    """An empty retrieval result should produce an empty context."""

    builder = ContextBuilder()
    context = builder.build([])

    assert context == ""


def test_build_returns_context_for_single_chunk() -> None:
    """A single chunk should be formatted correctly."""

    builder = ContextBuilder()
    context = builder.build(
        [
            create_retrieved_chunk("Hello world."),
        ]
    )

    assert context == ("Chunk 1\n\n" "Hello world.")


def test_build_preserves_chunk_order() -> None:
    """Retrieved chunks should appear in retrieval order."""

    builder = ContextBuilder()
    context = builder.build(
        [
            create_retrieved_chunk("First"),
            create_retrieved_chunk("Second"),
            create_retrieved_chunk("Third"),
        ]
    )

    assert context == (
        "Chunk 1\n\n" "First\n\n" "Chunk 2\n\n" "Second\n\n" "Chunk 3\n\n" "Third"
    )


def test_build_uses_only_chunk_text() -> None:
    """Only chunk text should appear in the generated context."""

    builder = ContextBuilder()
    context = builder.build(
        [
            create_retrieved_chunk(
                "Important content",
                distance=0.123,
            )
        ]
    )

    assert "Important content" in context
    assert "test.txt" not in context
    assert "0.123" not in context
