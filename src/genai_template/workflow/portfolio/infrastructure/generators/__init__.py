"""Official-SDK plain-text generation adapters."""

from genai_template.workflow.portfolio.infrastructure.generators.ollama import (
    OllamaTextGenerator,
)
from genai_template.workflow.portfolio.infrastructure.generators.openai import (
    OpenAITextGenerator,
)

__all__ = [
    "OllamaTextGenerator",
    "OpenAITextGenerator",
]
