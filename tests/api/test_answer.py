from unittest.mock import Mock

from fastapi.testclient import TestClient

from genai_template.api.dependencies import get_rag_service
from genai_template.api.main import app
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
    client: TestClient,
) -> None:
    """Verify the answer endpoint returns generated content.

    Args:
        client:
            FastAPI test client.

        app:
            FastAPI application instance.
    """

    mock_service = Mock(spec=RagService)
    mock_service.answer.return_value = RagResult(
        answer="Generated answer.",
        metrics=get_test_metrics(),
    )

    app.dependency_overrides[get_rag_service] = lambda: mock_service

    response = client.post(
        "/api/v1/answer",
        json={
            "query": "What is RAG?",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["answer"] == "Generated answer."
    assert body["metrics"]["retrieved_chunks"] == 2

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
        },
    )

    assert response.status_code == 422  # i.e. Unprocessable Entity
