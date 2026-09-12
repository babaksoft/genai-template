"""Unit tests for the Qdrant vector store."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import NAMESPACE_URL, uuid5

import pytest
from qdrant_client import models

from genai_template.common.types import VectorDistance
from genai_template.schemas import DocumentChunk
from genai_template.stores.vector import QdrantStore


def _chunk(
    chunk_id: str = "chunk/one",
    embedding: list[float] | None = None,
) -> DocumentChunk:
    """Build an embedded chunk for Qdrant unit tests.

    Args:
        chunk_id:
            Canonical chunk identifier.
        embedding:
            Optional embedding override.

    Returns:
        Canonical document chunk.
    """

    return DocumentChunk(
        id=chunk_id,
        document_id="document.md",
        text="Hello world.",
        metadata={"author": "Babak", "page": 3},
        embedding=[0.1, 0.2, 0.3] if embedding is None else embedding,
    )


@patch("genai_template.stores.vector.qdrant_store.QdrantClient")
def test_constructor_supports_local_mode(
    mock_client_class: MagicMock,
    tmp_path: Path,
) -> None:
    """Local mode should initialize Qdrant with its resolved path."""

    QdrantStore(
        collection_name="documents",
        distance=VectorDistance.COSINE,
        path=tmp_path,
    )

    mock_client_class.assert_called_once_with(path=str(tmp_path))


@patch("genai_template.stores.vector.qdrant_store.QdrantClient")
def test_constructor_supports_server_mode(mock_client_class: MagicMock) -> None:
    """Server mode should forward its URL and optional API key."""

    QdrantStore(
        collection_name="documents",
        distance=VectorDistance.COSINE,
        url="https://qdrant.example.com",
        api_key="secret",
    )

    mock_client_class.assert_called_once_with(
        url="https://qdrant.example.com",
        api_key="secret",
    )


@pytest.mark.parametrize(
    ("path", "url", "api_key"),
    [
        (None, None, None),
        (Path("qdrant"), "https://qdrant.example.com", None),
        (Path("qdrant"), None, "secret"),
    ],
)
def test_constructor_rejects_invalid_connection_options(
    path: Path | None,
    url: str | None,
    api_key: str | None,
) -> None:
    """Connection options should describe exactly one valid mode."""

    with pytest.raises(ValueError):
        QdrantStore(
            collection_name="documents",
            distance=VectorDistance.COSINE,
            path=path,
            url=url,
            api_key=api_key,
        )


@pytest.mark.parametrize(
    ("distance", "qdrant_distance"),
    [
        (VectorDistance.COSINE, models.Distance.COSINE),
        (VectorDistance.INNER_PRODUCT, models.Distance.DOT),
        (VectorDistance.L2, models.Distance.EUCLID),
    ],
)
@patch("genai_template.stores.vector.qdrant_store.QdrantClient")
def test_first_upsert_creates_collection_and_round_trips_payload(
    mock_client_class: MagicMock,
    distance: VectorDistance,
    qdrant_distance: models.Distance,
    tmp_path: Path,
) -> None:
    """The first write should derive collection size and preserve chunk data."""

    client = mock_client_class.return_value
    client.collection_exists.return_value = False
    chunk = _chunk()
    store = QdrantStore(
        collection_name="documents",
        distance=distance,
        path=tmp_path,
    )

    store.upsert([chunk])

    client.create_collection.assert_called_once_with(
        collection_name="documents",
        vectors_config=models.VectorParams(size=3, distance=qdrant_distance),
    )
    points = client.upsert.call_args.kwargs["points"]
    assert len(points) == 1
    assert points[0].id == str(uuid5(NAMESPACE_URL, chunk.id))
    assert points[0].vector == chunk.embedding
    assert points[0].payload == {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "text": chunk.text,
        "metadata": chunk.metadata,
    }


@patch("genai_template.stores.vector.qdrant_store.QdrantClient")
def test_upsert_uses_existing_collection(mock_client_class: MagicMock) -> None:
    """Subsequent writes should not recreate an existing collection."""

    client = mock_client_class.return_value
    client.collection_exists.return_value = True
    store = QdrantStore(
        collection_name="documents",
        distance=VectorDistance.COSINE,
        url="https://qdrant.example.com",
    )

    store.upsert([_chunk()])

    client.create_collection.assert_not_called()
    client.upsert.assert_called_once()


@pytest.mark.parametrize(
    "chunks",
    [
        [
            DocumentChunk(
                id="missing",
                document_id="document.md",
                text="Missing.",
            )
        ],
        [_chunk(embedding=[])],
        [_chunk("one", [0.1, 0.2]), _chunk("two", [0.1, 0.2, 0.3])],
    ],
)
@patch("genai_template.stores.vector.qdrant_store.QdrantClient")
def test_upsert_validates_every_chunk_before_writing(
    mock_client_class: MagicMock,
    chunks: list[DocumentChunk],
) -> None:
    """Invalid embeddings should prevent collection and point mutations."""

    client = mock_client_class.return_value
    store = QdrantStore(
        collection_name="documents",
        distance=VectorDistance.COSINE,
        url="https://qdrant.example.com",
    )

    with pytest.raises(ValueError):
        store.upsert(chunks)

    client.collection_exists.assert_not_called()
    client.create_collection.assert_not_called()
    client.upsert.assert_not_called()


@patch("genai_template.stores.vector.qdrant_store.QdrantClient")
def test_empty_upsert_is_no_op(mock_client_class: MagicMock) -> None:
    """An empty upsert should not inspect or mutate Qdrant."""

    client = mock_client_class.return_value
    store = QdrantStore(
        collection_name="documents",
        distance=VectorDistance.COSINE,
        url="https://qdrant.example.com",
    )

    store.upsert([])

    client.collection_exists.assert_not_called()
    client.create_collection.assert_not_called()
    client.upsert.assert_not_called()


@pytest.mark.parametrize(
    ("distance", "scores", "expected"),
    [
        (VectorDistance.COSINE, [0.25, 0.9], [0.1, 0.75]),
        (VectorDistance.INNER_PRODUCT, [0.25, 0.9], [-0.9, -0.25]),
        (VectorDistance.L2, [0.75, 0.1], [0.1, 0.75]),
    ],
)
@patch("genai_template.stores.vector.qdrant_store.QdrantClient")
def test_search_normalizes_distances_and_orders_results(
    mock_client_class: MagicMock,
    distance: VectorDistance,
    scores: list[float],
    expected: list[float],
) -> None:
    """Search scores should become ascending provider-independent distances."""

    client = mock_client_class.return_value
    client.collection_exists.return_value = True
    client.query_points.return_value.points = [
        models.ScoredPoint(
            id=index,
            version=1,
            score=score,
            payload={
                "chunk_id": f"chunk-{index}",
                "document_id": "document.md",
                "text": f"Text {index}",
                "metadata": {"index": index},
            },
        )
        for index, score in enumerate(scores)
    ]
    store = QdrantStore(
        collection_name="documents",
        distance=distance,
        url="https://qdrant.example.com",
    )

    results = store.search([0.1, 0.2, 0.3], top_k=2, query="hello")

    assert [result.distance for result in results] == pytest.approx(expected)
    assert {result.chunk.id for result in results} == {"chunk-0", "chunk-1"}
    assert {result.chunk.metadata["index"] for result in results} == {0, 1}
    client.query_points.assert_called_once_with(
        collection_name="documents",
        query=[0.1, 0.2, 0.3],
        limit=2,
        with_payload=True,
    )


@patch("genai_template.stores.vector.qdrant_store.QdrantClient")
def test_missing_collection_operations_are_safe(mock_client_class: MagicMock) -> None:
    """Read and delete operations should be safe before collection creation."""

    client = mock_client_class.return_value
    client.collection_exists.return_value = False
    store = QdrantStore(
        collection_name="documents",
        distance=VectorDistance.COSINE,
        url="https://qdrant.example.com",
    )

    assert store.count() == 0
    assert store.search([0.1, 0.2, 0.3], top_k=2) == []
    store.delete()

    client.count.assert_not_called()
    client.query_points.assert_not_called()
    client.delete_collection.assert_not_called()


@patch("genai_template.stores.vector.qdrant_store.QdrantClient")
def test_count_and_delete_existing_collection(mock_client_class: MagicMock) -> None:
    """Existing collection lifecycle calls should be delegated to Qdrant."""

    client = mock_client_class.return_value
    client.collection_exists.return_value = True
    client.count.return_value.count = 7
    store = QdrantStore(
        collection_name="documents",
        distance=VectorDistance.COSINE,
        url="https://qdrant.example.com",
    )

    assert store.count() == 7
    store.delete()

    client.count.assert_called_once_with(collection_name="documents", exact=True)
    client.delete_collection.assert_called_once_with(collection_name="documents")


@patch("genai_template.stores.vector.qdrant_store.QdrantClient")
def test_provider_errors_propagate(mock_client_class: MagicMock) -> None:
    """Qdrant errors should be exposed to callers unchanged."""

    client = mock_client_class.return_value
    client.collection_exists.side_effect = RuntimeError("unavailable")
    store = QdrantStore(
        collection_name="documents",
        distance=VectorDistance.COSINE,
        url="https://qdrant.example.com",
    )

    with pytest.raises(RuntimeError, match="unavailable"):
        store.count()
