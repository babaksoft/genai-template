"""Tests for the registry-aware UI API client."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

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
        "answer": "Answer [S1] [S9]",
        "metrics": metrics_json(),
        "sources": [
            {
                "label": "S1",
                "chunk_id": "guide-001",
                "document_name": "guide.md",
                "section": "/Introduction/",
                "content": "Exact retrieved content.",
                "distance": 0.12,
                "cited": True,
            }
        ],
        "citation_warnings": [
            {
                "code": "unsupported_citation_labels",
                "message": "The answer references an unavailable source.",
                "labels": ["S9"],
            }
        ],
    }

    response = ApiClient("http://localhost:8000").answer("Question", 2, 3)

    assert response.answer == "Answer [S1] [S9]"
    assert response.sources[0].document_name == "guide.md"
    assert response.sources[0].cited is True
    assert response.citation_warnings[0].labels == ["S9"]
    assert mock_post.call_args.kwargs["json"] == {
        "query": "Question",
        "experiment_id": 2,
        "rag_config_id": 3,
    }


@pytest.mark.parametrize("missing_field", ["sources", "citation_warnings"])
@patch("genai_template.ui.api_client.post")
def test_answer_requires_citation_fields(
    mock_post: MagicMock, missing_field: str
) -> None:
    """Answer responses must contain both citation contract fields.

    Args:
        mock_post:
            Mocked HTTP POST function.
        missing_field:
            Required citation field omitted from the response.
    """

    payload = {
        "answer": "Answer",
        "metrics": metrics_json(),
        "sources": [],
        "citation_warnings": [],
    }
    del payload[missing_field]
    mock_post.return_value.json.return_value = payload

    with pytest.raises(ValidationError):
        ApiClient("http://localhost:8000").answer("Question", 2, 3)


@patch("genai_template.ui.api_client.get")
def test_list_experiments(mock_get: MagicMock) -> None:
    """The UI client should deserialize experiment registry."""

    timestamp = datetime(2026, 9, 13, tzinfo=UTC)
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

    mock_get.side_effect = [experiment_response]
    client = ApiClient("http://localhost:8000")

    experiments = client.list_experiments()

    assert experiments[0].id == 2


@patch("genai_template.ui.api_client.get")
def test_list_configs(mock_get: MagicMock) -> None:
    """The UI client should deserialize RAG config registry."""

    timestamp = datetime(2026, 9, 13, tzinfo=UTC).isoformat()
    config = load_rag_config()
    config_response = MagicMock()
    config_response.json.return_value = [
        {
            "id": 3,
            "config_fingerprint": "a" * 64,
            "config": config.model_dump(mode="json"),
            "created_at": timestamp,
        }
    ]
    mock_get.side_effect = [config_response]
    client = ApiClient("http://localhost:8000")

    configs = client.list_rag_configs()

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


@patch("genai_template.ui.api_client.get")
def test_get_experiment_by_id(mock_get: MagicMock) -> None:
    """The UI client should expose experiment lookup by identifier."""

    timestamp = datetime(2026, 9, 13, tzinfo=UTC)
    experiment_response = MagicMock()
    experiment_response.json.return_value = {
        "id": 2,
        "source_id": 1,
        "name": "Trial",
        "description": None,
        "created_at": timestamp,
    }

    mock_get.side_effect = [experiment_response]
    client = ApiClient("http://localhost:8000")

    experiment = client.get_experiment(2)

    assert experiment.id == 2
    assert mock_get.call_args_list[0].args[0].endswith("/experiments/2")


@patch("genai_template.ui.api_client.get")
def test_get_rag_config_by_id(mock_get: MagicMock) -> None:
    """The UI client should expose RAG config lookup by identifier."""

    timestamp = datetime(2026, 9, 13, tzinfo=UTC)
    config = load_rag_config()
    config_response = MagicMock()
    config_response.json.return_value = {
        "id": 3,
        "config_fingerprint": "a" * 64,
        "config": config.model_dump(mode="json"),
        "created_at": timestamp,
    }

    mock_get.side_effect = [config_response]
    client = ApiClient("http://localhost:8000")

    rag_config = client.get_rag_config(3)

    assert rag_config.id == 3
    assert mock_get.call_args_list[0].args[0].endswith("/rag-configs/3")


@patch("genai_template.ui.api_client.post")
def test_register_rag_config_posts_resolved_config(mock_post: MagicMock) -> None:
    """The UI client should register the complete portable configuration."""

    timestamp = datetime(2026, 9, 13, tzinfo=UTC).isoformat()
    config = load_rag_config()
    mock_post.return_value.json.return_value = {
        "id": 9,
        "config_fingerprint": "b" * 64,
        "config": config.model_dump(mode="json"),
        "created_at": timestamp,
    }

    client = ApiClient("http://localhost:8000")
    result = client.register_rag_config(config)

    assert result.id == 9
    assert mock_post.call_args.kwargs["json"] == config.model_dump(mode="json")
