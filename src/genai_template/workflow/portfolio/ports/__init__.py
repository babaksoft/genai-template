"""Outbound ports used by Portfolio application services."""

from genai_template.workflow.portfolio.ports.repository import RepositoryReader
from genai_template.workflow.portfolio.ports.structured_generator import (
    StructuredGenerationResponse,
    StructuredSummaryGenerator,
)

__all__ = [
    "RepositoryReader",
    "StructuredGenerationResponse",
    "StructuredSummaryGenerator",
]
