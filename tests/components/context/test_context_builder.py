"""Tests for the ContextBuilder."""

from typing import Any

from genai_template.components.context import ContextBuilder
from genai_template.schemas import DocumentChunk, RetrievedChunk


def create_retrieved_chunk(
    text: str,
    *,
    chunk_id: str = "chunk-id",
    document_id: str = "document-id",
    metadata: dict[str, Any] | None = None,
    distance: float = 0.0,
) -> RetrievedChunk:
    """Create a retrieved chunk for testing.

    Args:
        text:
            Chunk content.
        chunk_id:
            Chunk identifier.
        document_id:
            Parent document identifier.
        metadata:
            Optional chunk metadata.
        distance:
            Retrieval distance.

    Returns:
        Retrieved chunk fixture.
    """

    return RetrievedChunk(
        chunk=DocumentChunk(
            id=chunk_id,
            document_id=document_id,
            text=text,
            metadata=metadata or {},
            embedding=None,
        ),
        distance=distance,
    )


def test_build_returns_empty_context_for_empty_input() -> None:
    """An empty retrieval result should produce an empty citation context."""

    context = ContextBuilder().build([])

    assert context.text == ""
    assert context.sources == []


def test_build_formats_source_metadata_and_markdown_section() -> None:
    """Context blocks should include safe document and section metadata."""

    context = ContextBuilder().build(
        [
            create_retrieved_chunk(
                "Important content",
                metadata={
                    "file_name": "/private/corpus/guide.md",
                    "file_path": "/private/corpus/guide.md",
                    "header_path": "/Introduction/Overview/",
                },
                distance=0.123,
            )
        ]
    )

    assert context.text == (
        "[S1]\n"
        "Document: guide.md\n"
        "Section: /Introduction/Overview/\n"
        "Content:\n"
        "Important content"
    )
    assert context.sources[0].model_dump() == {
        "label": "S1",
        "chunk_id": "chunk-id",
        "document_name": "guide.md",
        "section": "/Introduction/Overview/",
        "content": "Important content",
        "distance": 0.123,
        "cited": False,
    }
    assert "/private/corpus" not in context.text


def test_build_preserves_order_and_labels_duplicate_documents() -> None:
    """Each retrieved chunk should keep its order and receive a distinct label."""

    context = ContextBuilder().build(
        [
            create_retrieved_chunk(
                "First", chunk_id="one", metadata={"file_name": "guide.md"}
            ),
            create_retrieved_chunk(
                "Second", chunk_id="two", metadata={"file_name": "guide.md"}
            ),
            create_retrieved_chunk(
                "Third", chunk_id="three", metadata={"file_name": "other.md"}
            ),
        ]
    )

    assert [source.label for source in context.sources] == ["S1", "S2", "S3"]
    assert [source.chunk_id for source in context.sources] == ["one", "two", "three"]
    assert [source.document_name for source in context.sources] == [
        "guide.md",
        "guide.md",
        "other.md",
    ]
    assert context.text.index("First") < context.text.index("Second")
    assert context.text.index("Second") < context.text.index("Third")


def test_build_falls_back_to_safe_document_id_basename() -> None:
    """Missing file-name metadata should safely fall back to document ID."""

    context = ContextBuilder().build(
        [
            create_retrieved_chunk("Unix", document_id="/private/docs/unix.md"),
            create_retrieved_chunk(
                "Windows", document_id=r"C:\private\docs\windows.md"
            ),
        ]
    )

    assert [source.document_name for source in context.sources] == [
        "unix.md",
        "windows.md",
    ]
    assert all(source.section is None for source in context.sources)
    assert "/private/docs" not in context.text
    assert r"C:\private\docs" not in context.text
