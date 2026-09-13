"""Tests for application dependency composition."""

from unittest.mock import MagicMock, patch

from genai_template.api.dependencies import (
    get_rag_service,
    get_source_service,
)


@patch("genai_template.api.dependencies.RagService")
@patch("genai_template.api.dependencies.SourceService")
@patch("genai_template.api.dependencies.RagConfigService")
@patch("genai_template.api.dependencies.ExperimentService")
@patch("genai_template.api.dependencies.PromptBuilder")
@patch("genai_template.api.dependencies.ContextBuilder")
def test_get_rag_service_injects_registries(
    mock_context_builder: MagicMock,
    mock_prompt_builder: MagicMock,
    mock_experiment_service: MagicMock,
    mock_rag_config_service: MagicMock,
    mock_source_service: MagicMock,
    mock_rag_service: MagicMock,
) -> None:
    """The RAG dependency should compose dynamic execution registries."""

    result = get_rag_service()

    assert result is mock_rag_service.return_value
    assert (
        mock_source_service.call_args.kwargs["rag_config_service"]
        is mock_rag_config_service.return_value
    )
    mock_experiment_service.assert_called_once()
    mock_rag_service.assert_called_once_with(
        context_builder=mock_context_builder.return_value,
        prompt_builder=mock_prompt_builder.return_value,
        experiment_service=mock_experiment_service.return_value,
        rag_config_service=mock_rag_config_service.return_value,
        source_service=mock_source_service.return_value,
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
