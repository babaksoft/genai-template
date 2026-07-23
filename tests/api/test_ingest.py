from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient

from genai_template.api.dependencies import get_indexing_pipeline
from genai_template.api.main import app
from genai_template.pipeline import IndexingPipeline
from genai_template.schemas import IndexingResult


def test_ingest_returns_indexing_summary(
    client: TestClient,
) -> None:
    """Verify the ingest endpoint returns an indexing summary.

    Args:
        client:
            FastAPI test client.

        app:
            FastAPI application instance.
    """

    mock_pipeline = Mock(spec=IndexingPipeline)
    mock_pipeline.run.return_value = IndexingResult(
        documents_indexed=2,
        chunks_indexed=18,
        indexing_time=0.37,
    )

    app.dependency_overrides[get_indexing_pipeline] = lambda: mock_pipeline
    response = client.post(
        "/api/v1/ingest",
        json={
            "directory": "data/documents",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body == {
        "documents_indexed": 2,
        "chunks_indexed": 18,
        "indexing_time": 0.37,
    }

    mock_pipeline.run.assert_called_once_with(Path("data/documents"))

    app.dependency_overrides.clear()


def test_ingest_rejects_empty_directory(
    client: TestClient,
) -> None:
    """Verify an empty directory path is rejected.

    Args:
        client:
            FastAPI test client.
    """

    response = client.post(
        "/api/v1/ingest",
        json={
            "directory": "",
        },
    )

    assert response.status_code == 422
