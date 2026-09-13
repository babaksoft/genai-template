"""Tests for application dependency composition."""

from unittest.mock import MagicMock, patch

from genai_template.api.dependencies import (
    get_rag_config,
    get_rag_service,
    get_source_service,
)
from genai_template.config import VectorStoreConfig


def test_get_rag_config_loads_application_defaults() -> None:
    """The API should expose a fully resolved settings-backed configuration."""

    config = get_rag_config()

    assert isinstance(config.vector_store, VectorStoreConfig)
    assert config.vector_store.persist_directory.is_absolute()


@patch("genai_template.api.dependencies.RagService")
@patch("genai_template.api.dependencies.SourceService")
@patch("genai_template.api.dependencies.RagConfigService")
@patch("genai_template.api.dependencies.ExperimentService")
@patch("genai_template.api.dependencies.create_llm")
@patch("genai_template.api.dependencies.PromptBuilder")
@patch("genai_template.api.dependencies.ContextBuilder")
def test_get_rag_service_injects_default_config(
    mock_context_builder: MagicMock,
    mock_prompt_builder: MagicMock,
    mock_create_llm: MagicMock,
    mock_experiment_service: MagicMock,
    mock_rag_config_service: MagicMock,
    mock_source_service: MagicMock,
    mock_rag_service: MagicMock,
) -> None:
    """The RAG dependency should compose every service with one config."""

    config = get_rag_config()

    result = get_rag_service(config)

    assert result is mock_rag_service.return_value
    mock_create_llm.assert_called_once_with(config.llm)
    assert (
        mock_source_service.call_args.kwargs["rag_config_service"]
        is mock_rag_config_service.return_value
    )
    mock_experiment_service.assert_called_once()
    mock_rag_service.assert_called_once_with(
        context_builder=mock_context_builder.return_value,
        prompt_builder=mock_prompt_builder.return_value,
        language_model=mock_create_llm.return_value,
        experiment_service=mock_experiment_service.return_value,
        source_service=mock_source_service.return_value,
        config=config,
    )


@patch("genai_template.api.dependencies.SourceService")
@patch("genai_template.api.dependencies.RagConfigService")
def test_get_source_service_injects_config_registry(
    mock_rag_config_service: MagicMock,
    mock_source_service: MagicMock,
) -> None:
    """The source dependency should share a persisted config registry."""

    result = get_source_service()

    assert result is mock_source_service.return_value
    assert (
        mock_source_service.call_args.kwargs["rag_config_service"]
        is mock_rag_config_service.return_value
    )
