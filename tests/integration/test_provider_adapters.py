"""Integration checks for the OpenAI and Qdrant provider adapters."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest
from llama_index.core import Document

from genai_template.common.types import VectorDistance
from genai_template.components.embeddings import OpenAIEmbeddingModel
from genai_template.components.language_models import OpenAILanguageModel
from genai_template.components.splitters import MarkdownDocumentSplitter
from genai_template.pipelines import RetrievalPipeline
from genai_template.schemas import DocumentChunk
from genai_template.stores.vector import QdrantStore

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
OPENAI_REQUIRED = pytest.mark.skipif(
    not OPENAI_API_KEY,
    reason="OPENAI_API_KEY is required for this integration test.",
)


def _embedded_chunk(
    chunk_id: str,
    text: str,
    embedding: list[float],
) -> DocumentChunk:
    """Build an embedded chunk for a vector-store lifecycle check.

    Args:
        chunk_id:
            Canonical chunk identifier.
        text:
            Chunk contents.
        embedding:
            Dense vector to store.

    Returns:
        Embedded canonical document chunk.
    """

    return DocumentChunk(
        id=chunk_id,
        document_id="integration.md",
        text=text,
        metadata={"source": "integration"},
        embedding=embedding,
    )


@pytest.mark.integration
def test_local_qdrant_lifecycle(tmp_path: Path) -> None:
    """Local Qdrant should support create, upsert, search, and delete."""

    store = QdrantStore(
        collection_name=f"local-integration-{uuid4().hex}",
        distance=VectorDistance.COSINE,
        path=tmp_path / "qdrant",
    )
    chunks = [
        _embedded_chunk("near", "Nearest chunk.", [1.0, 0.0, 0.0]),
        _embedded_chunk("far", "Farther chunk.", [0.0, 1.0, 0.0]),
    ]

    assert store.count() == 0
    store.upsert(chunks)

    assert store.count() == 2
    results = store.search([1.0, 0.0, 0.0], top_k=2)
    assert [result.chunk.id for result in results] == ["near", "far"]
    assert results[0].chunk.metadata == {"source": "integration"}

    store.delete()
    assert store.count() == 0


@pytest.mark.integration
@OPENAI_REQUIRED
def test_openai_embedding_smoke() -> None:
    """OpenAI should generate document and query embeddings."""

    embedder = OpenAIEmbeddingModel(
        model_name="text-embedding-3-small",
        dimensions=256,
    )
    chunks = [
        DocumentChunk(
            id="openai-smoke-000",
            document_id="openai-smoke.md",
            text="Paris is the capital of France.",
        )
    ]

    embedded = embedder.embed(chunks)
    query_embedding = embedder.embed_query("What is the capital of France?")

    assert len(embedded[0].embedding or []) == 256
    assert len(query_embedding) == 256


@pytest.mark.integration
@OPENAI_REQUIRED
def test_openai_generation_smoke() -> None:
    """OpenAI should generate a non-empty response."""

    llm = OpenAILanguageModel(model_name="gpt-4o-mini")

    response = llm.generate("Reply with only the word: ready")

    assert response.strip()


@pytest.mark.integration
@pytest.mark.skipif(
    not QDRANT_URL,
    reason="QDRANT_URL is required for the server integration test.",
)
def test_qdrant_server_lifecycle() -> None:
    """A configured Qdrant server should support an isolated lifecycle."""

    store = QdrantStore(
        collection_name=f"server-integration-{uuid4().hex}",
        distance=VectorDistance.COSINE,
        url=QDRANT_URL,
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    try:
        store.upsert([_embedded_chunk("server", "Server chunk.", [1.0, 0.0])])

        assert store.count() == 1
        results = store.search([1.0, 0.0], top_k=1)
        assert results[0].chunk.id == "server"
    finally:
        store.delete()


@pytest.mark.integration
@OPENAI_REQUIRED
def test_markdown_openai_qdrant_composition(tmp_path: Path) -> None:
    """Markdown, OpenAI, and local Qdrant should compose for retrieval."""

    chunks = MarkdownDocumentSplitter().split(
        [
            Document(
                id_="guide.md",
                text=(
                    "# France\n\nParis is the capital of France.\n\n"
                    "# Germany\n\nBerlin is the capital of Germany."
                ),
                metadata={"file_name": "guide.md"},
            )
        ]
    )
    embedder = OpenAIEmbeddingModel(
        model_name="text-embedding-3-small",
        dimensions=256,
    )
    store = QdrantStore(
        collection_name=f"composition-{uuid4().hex}",
        distance=VectorDistance.COSINE,
        path=tmp_path / "qdrant",
    )

    store.upsert(embedder.embed(chunks))
    results = RetrievalPipeline(embedder=embedder, store=store, top_k=1).retrieve(
        "What is the capital of France?"
    )

    assert len(results) == 1
    assert "Paris" in results[0].chunk.text
