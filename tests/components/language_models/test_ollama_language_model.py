"""Tests for the OllamaLanguageModel."""

from unittest.mock import MagicMock, patch

import pytest

from genai_template.components.language_models import (
    OllamaLanguageModel,
)

LLM_MODEL = "llama3"


@patch("genai_template.components.language_models.ollama_language_model.Ollama")
def test_init_creates_ollama_instance(mock_ollama: MagicMock) -> None:
    """The adapter should create an Ollama instance."""

    OllamaLanguageModel(model_name=LLM_MODEL)

    mock_ollama.assert_called_once_with(model=LLM_MODEL, request_timeout=180)


@patch("genai_template.components.language_models.ollama_language_model.Ollama")
def test_generate_calls_complete(
    mock_ollama: MagicMock,
) -> None:
    """The adapter should call the underlying language model."""

    llm = mock_ollama.return_value

    response = MagicMock()
    response.text = "Paris"

    llm.complete.return_value = response

    model = OllamaLanguageModel(model_name=LLM_MODEL)
    answer = model.generate("What is the capital of France?")

    llm.complete.assert_called_once_with("What is the capital of France?")

    assert answer == "Paris"


@patch("genai_template.components.language_models.ollama_language_model.Ollama")
def test_generate_propagates_exceptions(
    mock_ollama: MagicMock,
) -> None:
    """Exceptions from the underlying language model should propagate."""

    llm = mock_ollama.return_value
    llm.complete.side_effect = RuntimeError("Connection failed")

    model = OllamaLanguageModel(model_name=LLM_MODEL)

    with pytest.raises(RuntimeError, match="Connection failed"):
        model.generate("Hello")
