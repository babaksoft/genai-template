"""Application service for parsed and validated summary generation."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import cast

from genai_template.workflow.portfolio.domain import (
    SUMMARY_SPECIFICATIONS,
    ComponentSummary,
    GenerationRequest,
    StructuredSummary,
    SummaryGenerationResponse,
)
from genai_template.workflow.portfolio.domain.contracts import TextGenerator
from genai_template.workflow.portfolio.infrastructure.generation.parser import (
    parse_summary_markdown,
)
from genai_template.workflow.portfolio.infrastructure.generation.validation import (
    validate_component_evidence,
    validate_project_evidence,
)


def generate_validated_summary[SummaryT: StructuredSummary](
    generator: TextGenerator,
    request: GenerationRequest,
    output_type: type[SummaryT],
    *,
    repository_context_paths: Iterable[str],
    component_evidence_paths: Iterable[str] = (),
) -> SummaryGenerationResponse[SummaryT]:
    """Generate text once, parse it, and enforce the exact evidence scope.

    Args:
        generator:
            Plain-text generation provider port.
        request:
            Fully identified prompt request.
        output_type:
            Strict expected output model.
        repository_context_paths:
            Exact repository paths supplied directly to the call.
        component_evidence_paths:
            Component evidence supplied to project-document synthesis.

    Returns:
        Parsed response containing the validated, scope-checked value.

    Raises:
        EvidenceValidationError:
            If parsed evidence falls outside the actual call inputs.
        TextGenerationError:
            If provider generation fails.
    """

    repository_paths = tuple(repository_context_paths)
    component_paths = tuple(component_evidence_paths)
    response = generator.generate(request)
    specification = SUMMARY_SPECIFICATIONS[request.provenance.artifact_kind]
    if specification.output_type is not output_type:
        raise ValueError("summary output type does not match artifact specification")
    parsed = parse_summary_markdown(
        response.text,
        specification,
        (*repository_paths, *component_paths),
    )
    value = cast(SummaryT, parsed.summary)
    if isinstance(value, ComponentSummary):
        validate_component_evidence(value, repository_paths)
    else:
        validate_project_evidence(value, repository_paths, component_paths)
    return SummaryGenerationResponse[SummaryT](
        value=value,
        provider=response.provider,
        model=response.model,
        token_usage=response.token_usage,
        provider_metadata=response.provider_metadata,
        raw_response_hash=hashlib.sha256(response.text.encode("utf-8")).hexdigest(),
        warnings=parsed.warnings,
    )
