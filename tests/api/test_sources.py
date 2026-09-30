"""Tests for corpus source and deterministic index API routes."""

from datetime import UTC, datetime
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from genai_template.api.dependencies import get_source_service
from genai_template.db.models import Source
from genai_template.schemas import (
    IndexBuildStatus,
    IndexingResult,
    IndexStatus,
    IndexStatusReason,
)
from genai_template.services import IndexBuildInProgressError, SourceService


def create_index_status(
    reason: IndexStatusReason = IndexStatusReason.CURRENT,
) -> IndexStatus:
    """Create a complete API index status fixture.

    Args:
        reason:
            Availability reason to represent.

    Returns:
        Typed status fixture.
    """

    timestamp = datetime(2026, 9, 30, tzinfo=UTC)
    return IndexStatus(
        source_id=4,
        rag_config_id=7,
        collection_name="idx-test",
        index_fingerprint="a" * 64,
        current_corpus_fingerprint="b" * 64,
        built_corpus_fingerprint="b" * 64,
        latest_build_id=11,
        latest_build_status=IndexBuildStatus.SUCCEEDED,
        build_started_at=timestamp,
        build_finished_at=timestamp,
        document_count=3,
        chunk_count=12,
        collection_count=12,
        indexing_duration=0.4,
        available=reason == IndexStatusReason.CURRENT,
        reason=reason,
    )


def test_list_source_candidates(app: FastAPI) -> None:
    """The candidates endpoint should return browsable corpus directories."""

    source_service = Mock(spec=SourceService)
    source_service.list_candidates.return_value = ["handbook", "product-docs"]
    app.dependency_overrides[get_source_service] = lambda: source_service

    response = TestClient(app).get("/api/v1/sources/candidates")

    assert response.status_code == 200
    assert response.json() == [{"name": "handbook"}, {"name": "product-docs"}]
    app.dependency_overrides.clear()


def test_list_sources_exposes_registration_metadata_only(app: FastAPI) -> None:
    """Source responses should not expose collection or indexing metadata."""

    source = Source(
        id=4,
        name="product-docs",
        directory="/corpora/product-docs",
        created_at=datetime(2026, 9, 2, tzinfo=UTC),
    )
    source_service = Mock(spec=SourceService)
    source_service.list_sources.return_value = [source]
    app.dependency_overrides[get_source_service] = lambda: source_service

    response = TestClient(app).get("/api/v1/sources")

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": 4,
            "name": "product-docs",
            "directory": "/corpora/product-docs",
            "created_at": "2026-09-02T00:00:00Z",
        }
    ]
    app.dependency_overrides.clear()


def test_create_source_registers_directory(app: FastAPI) -> None:
    """Source creation should register the directory without indexing it."""

    source = Source(
        id=4,
        name="product-docs",
        directory="/corpora/product-docs",
        created_at=datetime(2026, 9, 2, tzinfo=UTC),
    )
    source_service = Mock(spec=SourceService)
    source_service.register.return_value = source
    app.dependency_overrides[get_source_service] = lambda: source_service

    response = TestClient(app).post(
        "/api/v1/sources", json={"directory": "product-docs"}
    )

    assert response.status_code == 201
    assert response.json()["name"] == "product-docs"
    assert "documents_indexed" not in response.json()
    source_service.register.assert_called_once_with("product-docs")
    app.dependency_overrides.clear()


def test_create_source_rejects_duplicate_name(app: FastAPI) -> None:
    """Source creation should report duplicate source names."""

    source_service = Mock(spec=SourceService)
    source_service.register.side_effect = ValueError(
        "Source 'product-docs' already exists."
    )
    app.dependency_overrides[get_source_service] = lambda: source_service

    response = TestClient(app).post(
        "/api/v1/sources", json={"directory": "product-docs"}
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Source 'product-docs' already exists."
    app.dependency_overrides.clear()


def test_get_index_status_returns_typed_projection(app: FastAPI) -> None:
    """The GET index endpoint should expose the service freshness decision."""

    source_service = Mock(spec=SourceService)
    source_service.get_index_status.return_value = create_index_status()
    app.dependency_overrides[get_source_service] = lambda: source_service

    response = TestClient(app).get("/api/v1/sources/4/indexes/7")

    assert response.status_code == 200
    assert response.json()["reason"] == "current"
    assert response.json()["current_corpus_fingerprint"] == "b" * 64
    source_service.get_index_status.assert_called_once_with(4, 7)
    app.dependency_overrides.clear()


def test_rebuild_index_returns_persisted_build_and_status(app: FastAPI) -> None:
    """The PUT endpoint should retain metrics and add durable build state."""

    source_service = Mock(spec=SourceService)
    source_service.rebuild_index.return_value = IndexingResult(
        documents_indexed=3, chunks_indexed=12, indexing_time=0.4
    )
    source_service.get_index_status.return_value = create_index_status()
    app.dependency_overrides[get_source_service] = lambda: source_service

    response = TestClient(app).put("/api/v1/sources/4/indexes/7")

    assert response.status_code == 200
    assert response.json()["documents_indexed"] == 3
    assert response.json()["chunks_indexed"] == 12
    assert response.json()["indexing_time"] == 0.4
    assert response.json()["build"]["id"] == 11
    assert response.json()["build"]["status"] == "succeeded"
    assert response.json()["status"]["reason"] == "current"
    source_service.rebuild_index.assert_called_once_with(4, 7)
    app.dependency_overrides.clear()


def test_rebuild_index_reports_concurrent_attempt_as_conflict(app: FastAPI) -> None:
    """A process-local rebuild conflict should return an actionable reason."""

    source_service = Mock(spec=SourceService)
    source_service.rebuild_index.side_effect = IndexBuildInProgressError(
        "Index rebuild already in progress."
    )
    app.dependency_overrides[get_source_service] = lambda: source_service

    response = TestClient(app).put("/api/v1/sources/4/indexes/7")

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "message": "Index rebuild already in progress.",
        "reason": "building",
        "action": "retry_status",
        "status_endpoint": "/sources/4/indexes/7",
    }
    app.dependency_overrides.clear()


def test_rebuild_index_rejects_missing_source_or_config(app: FastAPI) -> None:
    """Missing registry records should be exposed as not-found responses."""

    source_service = Mock(spec=SourceService)
    source_service.rebuild_index.side_effect = ValueError(
        "RAG config 7 does not exist."
    )
    app.dependency_overrides[get_source_service] = lambda: source_service

    response = TestClient(app).put("/api/v1/sources/4/indexes/7")

    assert response.status_code == 404
    assert response.json()["detail"] == "RAG config 7 does not exist."
    app.dependency_overrides.clear()
