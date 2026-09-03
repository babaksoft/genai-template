"""Tests for corpus source API routes."""

from datetime import UTC, datetime
from unittest.mock import Mock

from fastapi.testclient import TestClient

from genai_template.api.dependencies import get_source_service
from genai_template.api.main import app
from genai_template.db.models import Source
from genai_template.services import SourceService


def test_list_source_candidates(client: TestClient) -> None:
    """The candidates endpoint should return browsable corpus directories."""

    source_service = Mock(spec=SourceService)
    source_service.list_candidates.return_value = ["handbook", "product-docs"]
    app.dependency_overrides[get_source_service] = lambda: source_service

    response = client.get("/api/v1/sources/candidates")

    assert response.status_code == 200
    assert response.json() == [{"name": "handbook"}, {"name": "product-docs"}]
    app.dependency_overrides.clear()


def test_list_sources(client: TestClient) -> None:
    """The sources endpoint should serialize persisted source metadata."""

    source = Source(
        id=4,
        name="product-docs",
        directory="/corpora/product-docs",
        collection_name="source-abc",
        documents_indexed=2,
        chunks_indexed=8,
        indexed_at=datetime(2026, 9, 2, tzinfo=UTC),
        indexing_time=0.3,
    )
    source_service = Mock(spec=SourceService)
    source_service.list_sources.return_value = [source]
    app.dependency_overrides[get_source_service] = lambda: source_service

    response = client.get("/api/v1/sources")

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": 4,
            "name": "product-docs",
            "directory": "/corpora/product-docs",
            "documents_indexed": 2,
            "chunks_indexed": 8,
            "indexed_at": "2026-09-02T00:00:00Z",
            "indexing_time": 0.3,
        }
    ]
    app.dependency_overrides.clear()


def test_ingest_source_returns_source(client: TestClient) -> None:
    """The source ingestion endpoint should return persisted source metadata."""

    source = Source(
        id=4,
        name="product-docs",
        directory="/corpora/product-docs",
        collection_name="source-abc",
        documents_indexed=2,
        chunks_indexed=8,
        indexed_at=datetime(2026, 9, 2, tzinfo=UTC),
        indexing_time=0.3,
    )
    source_service = Mock(spec=SourceService)
    source_service.ingest.return_value = source
    app.dependency_overrides[get_source_service] = lambda: source_service

    response = client.post("/api/v1/sources", json={"directory": "product-docs"})

    assert response.status_code == 201
    assert response.json()["name"] == "product-docs"
    assert response.json()["documents_indexed"] == 2
    assert response.json()["chunks_indexed"] == 8
    assert response.json()["indexing_time"] == 0.3

    source_service.ingest.assert_called_once_with("product-docs")
    app.dependency_overrides.clear()


def test_ingest_source_rejects_duplicate_name(client: TestClient) -> None:
    """The source ingestion endpoint should report duplicate source names."""

    source_service = Mock(spec=SourceService)
    source_service.ingest.side_effect = ValueError(
        "Source 'product-docs' already exists."
    )
    app.dependency_overrides[get_source_service] = lambda: source_service

    response = client.post("/api/v1/sources", json={"directory": "product-docs"})

    assert response.status_code == 409
    assert response.json()["detail"] == "Source 'product-docs' already exists."
    app.dependency_overrides.clear()
