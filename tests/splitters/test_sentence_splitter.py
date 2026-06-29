"""Unit tests for the document splitter."""

from llama_index.core import Document

from genai_template.components.splitters.sentence_splitter import (
    DocumentSplitter,
)


def test_split_single_document() -> None:
    """A document should be split into chunks."""

    document = Document(
        text="Sentence one. " * 500,
        metadata={
            "file_name": "document.md",
        },
    )

    splitter = DocumentSplitter()

    chunks = splitter.split([document])

    assert len(chunks) > 1


def test_empty_input() -> None:
    """Empty input should return an empty list."""

    splitter = DocumentSplitter()

    assert splitter.split([]) == []


def test_metadata_preserved() -> None:
    """Metadata should be preserved."""

    document = Document(
        text="Hello world.",
        metadata={
            "file_name": "document.md",
            "author": "Babak",
        },
    )

    splitter = DocumentSplitter()

    chunk = splitter.split([document])[0]

    assert chunk.metadata["author"] == "Babak"


def test_embedding_is_none() -> None:
    """Embedding should not yet exist."""

    document = Document(
        text="Hello world.",
        metadata={
            "file_name": "document.md",
        },
    )

    splitter = DocumentSplitter()

    chunk = splitter.split([document])[0]

    assert chunk.embedding is None


def test_chunk_ids_are_deterministic() -> None:
    """Chunk IDs should follow the expected pattern."""

    document = Document(
        text="Sentence. " * 500,
        metadata={
            "file_name": "document.md",
        },
    )

    splitter = DocumentSplitter()

    chunks = splitter.split([document])

    for index, chunk in enumerate(chunks):
        assert chunk.id == f"document.md-{index:03d}"
