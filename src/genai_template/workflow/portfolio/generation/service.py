"""Application service for validated structured summary generation."""

from __future__ import annotations

from collections.abc import Iterable

from genai_template.workflow.portfolio.domain.generation import GenerationRequest
from genai_template.workflow.portfolio.domain.summaries import (
    ComponentSummary,
    StructuredSummary,
)
from genai_template.workflow.portfolio.generation.validation import (
    validate_component_evidence,
    validate_project_evidence,
)
from genai_template.workflow.portfolio.ports.structured_generator import (
    StructuredGenerationResponse,
    StructuredSummaryGenerator,
)


def generate_validated_summary[SummaryT: StructuredSummary](
    generator: StructuredSummaryGenerator,
    request: GenerationRequest,
    output_type: type[SummaryT],
    *,
    repository_context_paths: Iterable[str],
    component_evidence_paths: Iterable[str] = (),
) -> StructuredGenerationResponse[SummaryT]:
    """Generate a typed summary and enforce its exact evidence scope.

    Args:
        generator:
            Structured-generation provider port.
        request:
            Fully identified prompt request.
        output_type:
            Strict expected output model.
        repository_context_paths:
            Exact repository paths supplied directly to the call.
        component_evidence_paths:
            Component evidence supplied to project-document synthesis.

    Returns:
        Provider response containing the validated, scope-checked value.

    Raises:
        EvidenceValidationError:
            If generated evidence falls outside the actual call inputs.
        StructuredGenerationError:
            If provider generation or schema validation fails.
    """

    response = generator.generate(request, output_type)
    if isinstance(response.value, ComponentSummary):
        validate_component_evidence(response.value, repository_context_paths)
    else:
        validate_project_evidence(
            response.value,
            repository_context_paths,
            component_evidence_paths,
        )
    return response
