"""Factory for configured language models."""

from genai_template.components.language_models import OllamaLanguageModel
from genai_template.config.rag import LLMConfig


def create_llm(config: LLMConfig) -> OllamaLanguageModel:
    """Create a language model from validated configuration.

    Args:
        config:
            Language model configuration.

    Returns:
        Configured language model.

    Raises:
        ValueError:
            If the language model provider is unsupported.
    """

    if config.type == "ollama":
        return OllamaLanguageModel(
            model_name=config.model_name,
            base_url=config.base_url,
            request_timeout=config.request_timeout,
        )

    raise ValueError(f"Unsupported language model type: {config.type}")
