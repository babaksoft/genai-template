"""Unit tests for the FastEmbed embedding component."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from genai_template.components.embeddings import (
    FastEmbedEmbeddingModel,
)
from genai_template.schemas import DocumentChunk


def test_embed_populates_embeddings() -> None:
    """Embedding vectors should be attached to each chunk."""

    chunks = [
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

    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    with patch(
        "genai_template.components.embeddings.fastembed.FastEmbedEmbedding"
    ) as mock_embedding_class:
        mock_model = MagicMock()
        mock_model.get_text_embedding_batch.return_value = embeddings
        mock_embedding_class.return_value = mock_model

        embedder = FastEmbedEmbeddingModel()
        result = embedder.embed(chunks)

    assert result is chunks
    assert chunks[0].embedding == embeddings[0]
    assert chunks[1].embedding == embeddings[1]


def test_embed_empty_list() -> None:
    """Embedding an empty list should return an empty list."""

    with patch("genai_template.components.embeddings.fastembed.FastEmbedEmbedding"):
        embedder = FastEmbedEmbeddingModel()

    assert embedder.embed([]) == []


def test_chunk_order_is_preserved() -> None:
    """Embedding should preserve the original chunk order."""

    chunks = [
        DocumentChunk(
            id="chunk-001",
            document_id="document.md",
            text="Chunk A",
            metadata={},
        ),
        DocumentChunk(
            id="chunk-002",
            document_id="document.md",
            text="Chunk B",
            metadata={},
        ),
    ]

    embeddings = [
        [1.0],
        [2.0],
    ]

    with patch(
        "genai_template.components.embeddings.fastembed.FastEmbedEmbedding"
    ) as mock_embedding_class:
        mock_model = MagicMock()
        mock_model.get_text_embedding_batch.return_value = embeddings
        mock_embedding_class.return_value = mock_model

        embedder = FastEmbedEmbeddingModel()
        result = embedder.embed(chunks)

    assert result[0].id == "chunk-001"
    assert result[1].id == "chunk-002"


def test_batch_embedding_called_once() -> None:
    """The embedder should perform a single batch embedding call."""

    chunks = [
        DocumentChunk(
            id="chunk-001",
            document_id="document.md",
            text="First",
            metadata={},
        ),
        DocumentChunk(
            id="chunk-002",
            document_id="document.md",
            text="Second",
            metadata={},
        ),
    ]

    with patch(
        "genai_template.components.embeddings.fastembed.FastEmbedEmbedding"
    ) as mock_embedding_class:
        mock_model = MagicMock()
        mock_model.get_text_embedding_batch.return_value = [
            [0.1],
            [0.2],
        ]
        mock_embedding_class.return_value = mock_model

        embedder = FastEmbedEmbeddingModel()
        embedder.embed(chunks)

    mock_model.get_text_embedding_batch.assert_called_once_with(
        texts=["First", "Second"],
    )


def test_embed_query_empty() -> None:
    """An empty query should be rejected."""

    with patch("genai_template.components.embeddings.fastembed.FastEmbedEmbedding"):
        embedder = FastEmbedEmbeddingModel()

    with pytest.raises(ValueError):
        embedder.embed_query("   ")


def test_embed_query() -> None:
    """A query embedding should be generated."""

    embedding = [0.1, 0.2, 0.3]

    with patch(
        "genai_template.components.embeddings.fastembed.FastEmbedEmbedding"
    ) as mock_embedding_class:
        mock_model = MagicMock()
        mock_model.get_query_embedding.return_value = embedding
        mock_embedding_class.return_value = mock_model

        embedder = FastEmbedEmbeddingModel()
        result = embedder.embed_query("What is RAG?")

    assert result == embedding

    mock_model.get_query_embedding.assert_called_once_with(
        "What is RAG?",
    )
