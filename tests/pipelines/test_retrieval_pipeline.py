"""Unit tests for the retrieval pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

from genai_template.pipelines import RetrievalPipeline
from genai_template.schemas import DocumentChunk, RetrievedChunk


def test_retrieve() -> None:
    """The retrieval pipeline should return retrieved chunks."""

    embedding = [0.1, 0.2, 0.3]

    retrieved_chunk = RetrievedChunk(
        chunk=DocumentChunk(
            id="chunk-001",
            document_id="document.md",
            text="FastAPI is a modern Python web framework.",
            metadata={},
        ),
        distance=0.08,
    )

    embedder = MagicMock()
    embedder.embed_query.return_value = embedding

    store = MagicMock()
    store.search.return_value = [retrieved_chunk]

    pipeline = RetrievalPipeline(
        embedder=embedder,
        store=store,
    )

    result = pipeline.retrieve(
        query="What is FastAPI?",
        top_k=3,
    )

    assert result == [retrieved_chunk]

    embedder.embed_query.assert_called_once_with(
        "What is FastAPI?",
    )

    store.search.assert_called_once_with(
        embedding=embedding,
        top_k=3,
        query="What is FastAPI?",
    )


def test_retrieve_uses_configured_top_k() -> None:
    """Retrieval should use the pipeline top-k when no override is supplied."""

    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1]
    store = MagicMock()
    store.search.return_value = []
    pipeline = RetrievalPipeline(embedder=embedder, store=store, top_k=7)

    pipeline.retrieve(query="Configured query")

    store.search.assert_called_once_with(
        embedding=[0.1],
        top_k=7,
        query="Configured query",
    )
