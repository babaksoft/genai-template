from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from genai_template.api.dependencies import get_rag_service
from genai_template.schemas import (
    CitationSource,
    CitationWarning,
    RagResult,
    RunMetrics,
)
from genai_template.services import IndexNotBuiltError, RagService


def get_test_metrics() -> RunMetrics:
    """Create representative API response metrics.

    Returns:
        Valid run metrics.
    """

    return RunMetrics(
        query="test",
        embedding_model="test",
        vector_store="test",
        llm_model="test",
        top_k=2,
        retrieved_chunks=2,
        best_distance=0.1,
        worst_distance=0.4,
        context_length=200,
        prompt_length=300,
        response_length=50,
        retrieval_time=0.01,
        generation_time=0.5,
        total_time=0.51,
    )


def test_answer_returns_generated_response(
    app: FastAPI,
) -> None:
    """Verify the answer endpoint returns generated content.

    Args:
        app:
            FastAPI application instance.
    """

    mock_service = Mock(spec=RagService)
    mock_service.answer.return_value = RagResult(
        answer="Generated answer [S1].",
        metrics=get_test_metrics(),
        sources=[
            CitationSource(
                label="S1",
                chunk_id="guide-001",
                document_name="guide.md",
                section="/Introduction/",
                content="Exact content.",
                distance=0.1,
                cited=True,
            )
        ],
        citation_warnings=[
            CitationWarning(
                code="unsupported_citation_labels",
                message="The answer references labels not present in its context.",
                labels=["S9"],
            )
        ],
    )

    app.dependency_overrides[get_rag_service] = lambda: mock_service

    client = TestClient(app)
    response = client.post(
        "/api/v1/answer",
        json={
            "query": "What is RAG?",
            "experiment_id": 1,
            "rag_config_id": 2,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body == {
        "answer": "Generated answer [S1].",
        "metrics": get_test_metrics().model_dump(mode="json"),
        "sources": [
            {
                "label": "S1",
                "chunk_id": "guide-001",
                "document_name": "guide.md",
                "section": "/Introduction/",
                "content": "Exact content.",
                "distance": 0.1,
                "cited": True,
            }
        ],
        "citation_warnings": [
            {
                "code": "unsupported_citation_labels",
                "message": "The answer references labels not present in its context.",
                "labels": ["S9"],
            }
        ],
    }
    mock_service.answer.assert_called_once_with("What is RAG?", 1, 2)

    app.dependency_overrides.clear()


def test_answer_rejects_empty_query(
    client: TestClient,
) -> None:
    """Verify empty queries are rejected.

    Args:
        client:
            FastAPI test client.
    """

    response = client.post(
        "/api/v1/answer",
        json={
            "query": "",
            "experiment_id": 1,
            "rag_config_id": 2,
        },
    )

    assert response.status_code == 422  # i.e. Unprocessable Entity


def test_answer_rejects_missing_experiment(app: FastAPI) -> None:
    """Verify unknown experiment identifiers return a not-found response.

    Args:
        app:
            FastAPI application instance.
    """

    mock_service = Mock(spec=RagService)
    mock_service.answer.side_effect = ValueError("Experiment 99 does not exist.")

    app.dependency_overrides[get_rag_service] = lambda: mock_service

    client = TestClient(app)
    response = client.post(
        "/api/v1/answer",
        json={"query": "What is RAG?", "experiment_id": 99, "rag_config_id": 2},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Experiment 99 does not exist."

    app.dependency_overrides.clear()


def test_answer_rejects_missing_index_with_conflict(app: FastAPI) -> None:
    """An unbuilt selected index should produce an actionable conflict."""

    mock_service = Mock(spec=RagService)
    mock_service.answer.side_effect = IndexNotBuiltError(
        "Index for source 1 and RAG config 2 has not been built."
    )
    app.dependency_overrides[get_rag_service] = lambda: mock_service

    response = TestClient(app).post(
        "/api/v1/answer",
        json={"query": "What is RAG?", "experiment_id": 1, "rag_config_id": 2},
    )

    assert response.status_code == 409
    assert "has not been built" in response.json()["detail"]

    app.dependency_overrides.clear()
