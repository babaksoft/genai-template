"""LlamaIndex-backed structured generation adapters."""

from genai_template.workflow.portfolio.adapters.generators.ollama import (
    OllamaStructuredSummaryGenerator,
)
from genai_template.workflow.portfolio.adapters.generators.openai import (
    OpenAIStructuredSummaryGenerator,
)

__all__ = [
    "OllamaStructuredSummaryGenerator",
    "OpenAIStructuredSummaryGenerator",
]
