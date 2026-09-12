"""Unit tests for the OpenAI language model adapter."""

from unittest.mock import MagicMock, patch

import pytest

from genai_template.components.language_models import OpenAILanguageModel


@patch("genai_template.components.language_models.openai_language_model.OpenAI")
def test_constructor_forwards_provider_options(mock_openai: MagicMock) -> None:
    """Model and timeout should reach the OpenAI adapter."""

    OpenAILanguageModel(model_name="gpt-4o-mini", request_timeout=30)

    mock_openai.assert_called_once_with(model="gpt-4o-mini", timeout=30)


@patch("genai_template.components.language_models.openai_language_model.OpenAI")
def test_generate_returns_response_text(mock_openai: MagicMock) -> None:
    """Generation should return normalized text from the provider response."""

    response = MagicMock()
    response.text = "Paris"
    mock_openai.return_value.complete.return_value = response
    model = OpenAILanguageModel("gpt-4o-mini")

    result = model.generate("What is the capital of France?")

    assert result == "Paris"
    mock_openai.return_value.complete.assert_called_once_with(
        "What is the capital of France?"
    )


@patch("genai_template.components.language_models.openai_language_model.OpenAI")
def test_generate_normalizes_non_string_text(mock_openai: MagicMock) -> None:
    """Provider response text should always be normalized to a string."""

    response = MagicMock()
    response.text = 42
    mock_openai.return_value.complete.return_value = response
    model = OpenAILanguageModel("gpt-4o-mini")

    assert model.generate("Answer briefly.") == "42"


@patch("genai_template.components.language_models.openai_language_model.OpenAI")
def test_generate_propagates_provider_errors(mock_openai: MagicMock) -> None:
    """Generation provider failures should propagate unchanged."""

    mock_openai.return_value.complete.side_effect = RuntimeError("OpenAI unavailable")
    model = OpenAILanguageModel("gpt-4o-mini")

    with pytest.raises(RuntimeError, match="OpenAI unavailable"):
        model.generate("Hello")
