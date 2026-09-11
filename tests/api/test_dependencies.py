"""Tests for application dependency composition."""

from unittest.mock import MagicMock, patch

from genai_template.api.dependencies import (
    get_rag_config,
    get_rag_service,
    get_source_service,
)


def test_get_rag_config_loads_application_defaults() -> None:
    """The API should expose a fully resolved settings-backed configuration."""

    config = get_rag_config()

    assert config.experiment.name
    assert config.vector_store.persist_directory.is_absolute()


@patch("genai_template.api.dependencies.create_rag_service")
def test_get_rag_service_injects_default_config(mock_create: MagicMock) -> None:
    """The RAG dependency should delegate composition with the supplied config."""

    config = get_rag_config()

    result = get_rag_service(config)

    assert result is mock_create.return_value
    mock_create.assert_called_once_with(config)


@patch("genai_template.api.dependencies.SourceService")
def test_get_source_service_injects_default_config(
    mock_source_service: MagicMock,
) -> None:
    """The source dependency should share the supplied RAG configuration."""

    config = get_rag_config()

    result = get_source_service(config)

    assert result is mock_source_service.return_value
    assert mock_source_service.call_args.kwargs["config"] is config
