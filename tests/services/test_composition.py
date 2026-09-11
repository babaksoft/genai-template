"""Tests for configured RAG service composition."""

from unittest.mock import MagicMock, patch

from genai_template.config import load_rag_config, settings
from genai_template.db import SessionLocal
from genai_template.services.composition import create_rag_service


@patch("genai_template.services.composition.RagService")
@patch("genai_template.services.composition.SourceService")
@patch("genai_template.services.composition.ExperimentService")
@patch("genai_template.services.composition.create_llm")
@patch("genai_template.services.composition.create_vector_store")
@patch("genai_template.services.composition.create_retrieval_pipeline")
@patch("genai_template.services.composition.create_embedder")
def test_create_rag_service_composes_configured_components(
    mock_create_embedder: MagicMock,
    mock_create_retrieval: MagicMock,
    mock_create_store: MagicMock,
    mock_create_llm: MagicMock,
    mock_experiment_service: MagicMock,
    mock_source_service: MagicMock,
    mock_rag_service: MagicMock,
) -> None:
    """The composition helper should pass one config through every component."""

    config = load_rag_config()

    result = create_rag_service(config)
    retrieval_factory = mock_rag_service.call_args.kwargs["retrieval_pipeline_factory"]
    retrieval = retrieval_factory("source-corpus")

    assert result is mock_rag_service.return_value
    assert retrieval is mock_create_retrieval.return_value
    mock_create_embedder.assert_called_once_with(config.embedder)
    mock_create_store.assert_called_once()
    store_config = mock_create_store.call_args.args[0]
    assert store_config == config.vector_store.model_copy(
        update={"collection_name": "source-corpus"}
    )
    mock_create_retrieval.assert_called_once_with(
        config.retrieval,
        mock_create_embedder.return_value,
        mock_create_store.return_value,
    )
    mock_create_llm.assert_called_once_with(config.llm)
    mock_source_service.assert_called_once_with(
        session_factory=SessionLocal,
        corpora_dir=settings.CORPORA_DIR,
        config=config,
    )
    assert mock_rag_service.call_args.kwargs["source_service"] is (
        mock_source_service.return_value
    )
    assert mock_rag_service.call_args.kwargs["config"] is config
    mock_experiment_service.assert_called_once()
