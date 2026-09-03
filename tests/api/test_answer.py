from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from genai_template.api.dependencies import get_rag_service
from genai_template.schemas import RagResult, RunMetrics
from genai_template.services import RagService


def get_test_metrics() -> RunMetrics:
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
        answer="Generated answer.",
        metrics=get_test_metrics(),
        retrieved_chunks=[],
    )

    app.dependency_overrides[get_rag_service] = lambda: mock_service

    client = TestClient(app)
    response = client.post(
        "/api/v1/answer",
        json={
            "query": "What is RAG?",
            "source_id": 1,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["answer"] == "Generated answer."
    assert body["metrics"]["retrieved_chunks"] == 2
    mock_service.answer.assert_called_once_with("What is RAG?", 1)

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
            "source_id": 1,
        },
    )

    assert response.status_code == 422  # i.e. Unprocessable Entity


def test_answer_rejects_missing_source(app: FastAPI) -> None:
    """Verify unknown source identifiers return a not-found response.

    Args:
        app:
            FastAPI application instance.
    """

    mock_service = Mock(spec=RagService)
    mock_service.answer.side_effect = ValueError("Source 99 does not exist.")

    app.dependency_overrides[get_rag_service] = lambda: mock_service

    client = TestClient(app)
    response = client.post(
        "/api/v1/answer",
        json={"query": "What is RAG?", "source_id": 99},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Source 99 does not exist."

    app.dependency_overrides.clear()
