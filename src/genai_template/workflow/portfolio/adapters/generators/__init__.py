"""Official-SDK plain-text generation adapters."""

from genai_template.workflow.portfolio.adapters.generators.ollama import (
    OllamaTextGenerator,
)
from genai_template.workflow.portfolio.adapters.generators.openai import (
    OpenAITextGenerator,
)

__all__ = [
    "OllamaTextGenerator",
    "OpenAITextGenerator",
]
