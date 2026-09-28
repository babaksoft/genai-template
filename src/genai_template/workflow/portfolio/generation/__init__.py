"""Structured portfolio summary generation."""

from genai_template.workflow.portfolio.domain.generation import (
    GenerationWarning,
    GenerationWarningCode,
)
from genai_template.workflow.portfolio.generation.components import (
    generate_component_summaries,
)
from genai_template.workflow.portfolio.generation.costs import estimate_generation_cost
from genai_template.workflow.portfolio.generation.parser import (
    ParsedSummary,
    parse_summary_markdown,
)
from genai_template.workflow.portfolio.generation.projects import (
    generate_project_summaries,
    resolve_project_context,
)
from genai_template.workflow.portfolio.generation.prompts import (
    ARCHITECTURE_PROMPT,
    COMPONENT_PROMPT,
    OVERVIEW_PROMPT,
    TESTING_OPERATIONS_PROMPT,
    PromptDefinition,
    assemble_component_prompt,
    assemble_project_prompt,
)
from genai_template.workflow.portfolio.generation.service import (
    generate_validated_summary,
)
from genai_template.workflow.portfolio.generation.specifications import (
    ARCHITECTURE_SPECIFICATION,
    COMPONENT_SPECIFICATION,
    OVERVIEW_SPECIFICATION,
    SUMMARY_SPECIFICATIONS,
    TESTING_OPERATIONS_SPECIFICATION,
    SummarySectionSpecification,
    SummarySpecification,
)
from genai_template.workflow.portfolio.generation.validation import (
    validate_component_evidence,
    validate_project_evidence,
)

__all__ = [
    "ARCHITECTURE_PROMPT",
    "ARCHITECTURE_SPECIFICATION",
    "COMPONENT_PROMPT",
    "COMPONENT_SPECIFICATION",
    "OVERVIEW_PROMPT",
    "OVERVIEW_SPECIFICATION",
    "SUMMARY_SPECIFICATIONS",
    "TESTING_OPERATIONS_PROMPT",
    "TESTING_OPERATIONS_SPECIFICATION",
    "GenerationWarning",
    "GenerationWarningCode",
    "ParsedSummary",
    "PromptDefinition",
    "SummarySectionSpecification",
    "SummarySpecification",
    "assemble_component_prompt",
    "assemble_project_prompt",
    "estimate_generation_cost",
    "generate_component_summaries",
    "generate_project_summaries",
    "generate_validated_summary",
    "parse_summary_markdown",
    "resolve_project_context",
    "validate_component_evidence",
    "validate_project_evidence",
]
