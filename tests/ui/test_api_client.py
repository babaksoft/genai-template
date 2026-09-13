"""Tests for the registry-aware UI API client."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from genai_template.config import load_rag_config
from genai_template.schemas import RunMetrics
from genai_template.ui.api_client import ApiClient


def metrics_json() -> dict[str, object]:
    """Create serialized answer metrics.

    Returns:
        JSON-compatible run metrics.
    """

    return RunMetrics(
        query="Question",
        embedding_model="embedder",
        vector_store="Chroma",
        llm_model="llm",
        top_k=5,
        retrieved_chunks=0,
        context_length=0,
        prompt_length=10,
        response_length=6,
        retrieval_time=0.1,
        generation_time=0.2,
        total_time=0.3,
    ).model_dump(mode="json")


@patch("genai_template.ui.api_client.post")
def test_answer_sends_experiment_and_config_ids(mock_post: MagicMock) -> None:
    """Answer requests should use canonical execution identifiers."""

    mock_post.return_value.json.return_value = {
        "answer": "Answer",
        "metrics": metrics_json(),
    }

    response = ApiClient("http://localhost:8000").answer("Question", 2, 3)

    assert response.answer == "Answer"
    assert mock_post.call_args.kwargs["json"] == {
        "query": "Question",
        "experiment_id": 2,
        "rag_config_id": 3,
    }


@patch("genai_template.ui.api_client.get")
def test_list_experiments_and_configs(mock_get: MagicMock) -> None:
    """The UI client should deserialize both selection registries."""

    timestamp = datetime(2026, 9, 13, tzinfo=UTC).isoformat()
    config = load_rag_config()
    experiment_response = MagicMock()
    experiment_response.json.return_value = [
        {
            "id": 2,
            "source_id": 1,
            "name": "Trial",
            "description": None,
            "created_at": timestamp,
        }
    ]
    config_response = MagicMock()
    config_response.json.return_value = [
        {
            "id": 3,
            "config_fingerprint": "a" * 64,
            "config": config.model_dump(mode="json"),
            "created_at": timestamp,
        }
    ]
    mock_get.side_effect = [experiment_response, config_response]
    client = ApiClient("http://localhost:8000")

    experiments = client.list_experiments()
    configs = client.list_rag_configs()

    assert experiments[0].id == 2
    assert configs[0].id == 3


@patch("genai_template.ui.api_client.put")
def test_rebuild_index_uses_put_endpoint(mock_put: MagicMock) -> None:
    """Index rebuilds should target the source/config deterministic route."""

    mock_put.return_value.json.return_value = {
        "documents_indexed": 2,
        "chunks_indexed": 8,
        "indexing_time": 0.4,
    }

    result = ApiClient("http://localhost:8000").rebuild_index(4, 7)

    assert result.chunks_indexed == 8
    assert mock_put.call_args.args[0].endswith("/sources/4/indexes/7")
