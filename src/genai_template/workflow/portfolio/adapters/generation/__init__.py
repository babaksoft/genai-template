"""LlamaIndex-backed structured generation adapters."""

from genai_template.workflow.portfolio.adapters.generation.ollama import (
    OllamaStructuredSummaryGenerator,
)
from genai_template.workflow.portfolio.adapters.generation.openai import (
    OpenAIStructuredSummaryGenerator,
)

__all__ = ["OllamaStructuredSummaryGenerator", "OpenAIStructuredSummaryGenerator"]
