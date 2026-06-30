"""Unit tests for the Chroma vector store."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from genai_template.config import settings
from genai_template.schemas.chunk import DocumentChunk
from genai_template.stores.vector.chroma_store import ChromaStore


@patch("genai_template.stores.vector.chroma_store.chromadb.PersistentClient")
def test_constructor(
    mock_client_class: MagicMock,
) -> None:
    """The vector store should initialize the Chroma collection."""

    mock_collection = MagicMock()
    mock_client = MagicMock()

    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client_class.return_value = mock_client

    ChromaStore()

    mock_client_class.assert_called_once_with(
        path=settings.CHROMA_PERSIST_DIR,
    )

    # Abstraction tradeoff: See ChatGPT message @ 2026-06-30 11:48 AM
    mock_client.get_or_create_collection.assert_called_once_with(
        name=settings.CHROMA_COLLECTION,
        metadata={
            "hnsw:space": "cosine",
        },
    )


@patch("genai_template.stores.vector.chroma_store.chromadb.PersistentClient")
def test_upsert(
    mock_client_class: MagicMock,
) -> None:
    """Embedded chunks should be persisted."""

    mock_collection = MagicMock()
    mock_client = MagicMock()

    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client_class.return_value = mock_client

    store = ChromaStore()

    chunk = DocumentChunk(
        id="chunk-001",
        document_id="document.md",
        text="Hello world.",
        metadata={"author": "Babak"},
        embedding=[0.1, 0.2, 0.3],
    )

    store.upsert([chunk])

    mock_collection.upsert.assert_called_once()

    kwargs = mock_collection.upsert.call_args.kwargs

    assert kwargs["ids"] == ["chunk-001"]
    assert kwargs["documents"] == ["Hello world."]
    assert kwargs["embeddings"] == [[0.1, 0.2, 0.3]]

    metadata = kwargs["metadatas"][0]

    assert metadata["document_id"] == "document.md"
    assert "metadata" in metadata


@patch("genai_template.stores.vector.chroma_store.chromadb.PersistentClient")
def test_upsert_empty_list(
    mock_client_class: MagicMock,
) -> None:
    """Persisting an empty list should be a no-op."""

    mock_collection = MagicMock()
    mock_client = MagicMock()

    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client_class.return_value = mock_client

    store = ChromaStore()

    store.upsert([])

    mock_collection.upsert.assert_not_called()


@patch("genai_template.stores.vector.chroma_store.chromadb.PersistentClient")
def test_upsert_missing_embedding(
    mock_client_class: MagicMock,
) -> None:
    """Chunks without embeddings should be rejected."""

    mock_collection = MagicMock()
    mock_client = MagicMock()

    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client_class.return_value = mock_client

    store = ChromaStore()

    chunk = DocumentChunk(
        id="chunk-001",
        document_id="document.md",
        text="Hello world.",
        metadata={},
    )

    with pytest.raises(ValueError):
        store.upsert([chunk])

    mock_collection.upsert.assert_not_called()


@patch("genai_template.stores.vector.chroma_store.chromadb.PersistentClient")
def test_search(
    mock_client_class: MagicMock,
) -> None:
    """Search should reconstruct retrieved chunks."""

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "ids": [["chunk-001"]],
        "documents": [["Hello world."]],
        "metadatas": [
            [
                {
                    "document_id": "document.md",
                    "metadata": '{"author": "Babak"}',
                }
            ]
        ],
        "distances": [[0.12]],
    }

    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client_class.return_value = mock_client

    store = ChromaStore()
    result = store.search(
        embedding=[0.1, 0.2, 0.3],
        top_k=1,
    )

    assert len(result) == 1

    retrieved = result[0]

    assert retrieved.chunk.id == "chunk-001"
    assert retrieved.chunk.document_id == "document.md"
    assert retrieved.chunk.text == "Hello world."
    assert retrieved.chunk.metadata == {
        "author": "Babak",
    }

    assert retrieved.distance == 0.12

    mock_collection.query.assert_called_once_with(
        query_embeddings=[[0.1, 0.2, 0.3]],
        n_results=1,
    )


@patch("genai_template.stores.vector.chroma_store.chromadb.PersistentClient")
def test_search_empty_result(
    mock_client_class: MagicMock,
) -> None:
    """Empty query results should return an empty list."""

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "ids": [],
    }

    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client_class.return_value = mock_client

    store = ChromaStore()

    assert (
        store.search(
            embedding=[0.1, 0.2, 0.3],
            top_k=5,
        )
        == []
    )
