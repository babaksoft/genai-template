"""Unit tests for DocumentChunk."""

from genai_template.schemas import DocumentChunk


def test_chunk_creation() -> None:
    """A chunk should be created successfully."""

    chunk = DocumentChunk(
        id="chunk-001",
        document_id="doc-001",
        text="Hello world.",
        metadata={
            "source": "document.md",
            "page": 1,
        },
    )

    assert chunk.id == "chunk-001"
    assert chunk.document_id == "doc-001"
    assert chunk.embedding is None


def test_chunk_accepts_embedding() -> None:
    """Embedding should be optional."""

    chunk = DocumentChunk(
        id="chunk-001",
        document_id="doc-001",
        text="Hello",
        metadata={},
        embedding=[0.1, 0.2, 0.3],
    )

    assert chunk.embedding == [0.1, 0.2, 0.3]


def test_metadata_is_preserved() -> None:
    """Metadata should remain unchanged."""

    metadata = {
        "file_name": "document.md",
        "author": "Babak",
        "section": "Introduction",
    }

    chunk = DocumentChunk(
        id="1",
        document_id="doc",
        text="Sample",
        metadata=metadata,
    )

    assert chunk.metadata == metadata


def test_serialization() -> None:
    """Chunk should serialize correctly."""

    chunk = DocumentChunk(
        id="1",
        document_id="doc",
        text="Sample",
        metadata={},
    )

    dumped = chunk.model_dump()

    assert dumped["id"] == "1"
    assert dumped["embedding"] is None
