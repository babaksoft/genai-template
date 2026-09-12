"""Unit tests for the OpenAI embedding component."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from genai_template.components.embeddings import OpenAIEmbeddingModel
from genai_template.schemas import DocumentChunk


def _chunks() -> list[DocumentChunk]:
    """Create document chunks for embedding tests.

    Returns:
        Two unembedded document chunks.
    """

    return [
        DocumentChunk(
            id="chunk-001",
            document_id="document.md",
            text="First chunk.",
            metadata={},
        ),
        DocumentChunk(
            id="chunk-002",
            document_id="document.md",
            text="Second chunk.",
            metadata={},
        ),
    ]


@patch("genai_template.components.embeddings.openai.OpenAIEmbedding")
def test_constructor_forwards_provider_options(mock_embedder: MagicMock) -> None:
    """Model, dimensions, and timeout should reach the OpenAI adapter."""

    OpenAIEmbeddingModel(
        model_name="text-embedding-3-small",
        dimensions=512,
        request_timeout=30,
    )

    mock_embedder.assert_called_once_with(
        model="text-embedding-3-small",
        dimensions=512,
        timeout=30,
    )


@patch("genai_template.components.embeddings.openai.OpenAIEmbedding")
def test_constructor_accepts_default_dimensions(mock_embedder: MagicMock) -> None:
    """Omitted dimensions should preserve the provider model default."""

    OpenAIEmbeddingModel(
        model_name="text-embedding-3-small",
        request_timeout=45,
    )

    mock_embedder.assert_called_once_with(
        model="text-embedding-3-small",
        dimensions=None,
        timeout=45,
    )


@patch("genai_template.components.embeddings.openai.OpenAIEmbedding")
def test_embed_batches_and_populates_chunks(mock_embedder: MagicMock) -> None:
    """Chunk texts should be batched and vectors attached in order."""

    chunks = _chunks()
    embeddings = [[0.1, 0.2], [0.3, 0.4]]
    mock_embedder.return_value.get_text_embedding_batch.return_value = embeddings

    result = OpenAIEmbeddingModel("text-embedding-3-small").embed(chunks)

    assert result is chunks
    assert [chunk.embedding for chunk in chunks] == embeddings
    mock_embedder.return_value.get_text_embedding_batch.assert_called_once_with(
        texts=["First chunk.", "Second chunk."],
    )


@patch("genai_template.components.embeddings.openai.OpenAIEmbedding")
def test_embed_empty_list_avoids_provider_call(mock_embedder: MagicMock) -> None:
    """An empty chunk list should return without requesting embeddings."""

    model = OpenAIEmbeddingModel("text-embedding-3-small")

    assert model.embed([]) == []
    mock_embedder.return_value.get_text_embedding_batch.assert_not_called()


@patch("genai_template.components.embeddings.openai.OpenAIEmbedding")
def test_embed_query_uses_query_endpoint(mock_embedder: MagicMock) -> None:
    """Queries should use the provider's query embedding operation."""

    mock_embedder.return_value.get_query_embedding.return_value = [0.1, 0.2]
    model = OpenAIEmbeddingModel("text-embedding-3-small")

    result = model.embed_query("What is RAG?")

    assert result == [0.1, 0.2]
    mock_embedder.return_value.get_query_embedding.assert_called_once_with(
        "What is RAG?"
    )


@patch("genai_template.components.embeddings.openai.OpenAIEmbedding")
def test_embed_query_rejects_blank_input(mock_embedder: MagicMock) -> None:
    """A blank query should be rejected before calling OpenAI."""

    model = OpenAIEmbeddingModel("text-embedding-3-small")

    with pytest.raises(ValueError, match="Query must not be empty"):
        model.embed_query("   ")

    mock_embedder.return_value.get_query_embedding.assert_not_called()


@patch("genai_template.components.embeddings.openai.OpenAIEmbedding")
def test_embed_propagates_provider_errors(mock_embedder: MagicMock) -> None:
    """Embedding provider failures should propagate unchanged."""

    mock_embedder.return_value.get_text_embedding_batch.side_effect = RuntimeError(
        "OpenAI unavailable"
    )
    model = OpenAIEmbeddingModel("text-embedding-3-small")

    with pytest.raises(RuntimeError, match="OpenAI unavailable"):
        model.embed(_chunks())


@patch("genai_template.components.embeddings.openai.OpenAIEmbedding")
def test_embed_query_propagates_provider_errors(mock_embedder: MagicMock) -> None:
    """Query embedding provider failures should propagate unchanged."""

    mock_embedder.return_value.get_query_embedding.side_effect = RuntimeError(
        "OpenAI unavailable"
    )
    model = OpenAIEmbeddingModel("text-embedding-3-small")

    with pytest.raises(RuntimeError, match="OpenAI unavailable"):
        model.embed_query("What is RAG?")
