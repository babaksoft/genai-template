"""Application service for parsed and validated summary generation."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import cast

from pydantic import Field

from genai_template.workflow.portfolio.domain.generation import (
    GenerationRequest,
    GenerationWarning,
    ProviderAuditMetadata,
    TokenUsage,
)
from genai_template.workflow.portfolio.domain.snapshot import _ImmutableDomainModel
from genai_template.workflow.portfolio.domain.summaries import (
    ComponentSummary,
    StructuredSummary,
)
from genai_template.workflow.portfolio.generation.parser import parse_summary_markdown
from genai_template.workflow.portfolio.generation.specifications import (
    SUMMARY_SPECIFICATIONS,
)
from genai_template.workflow.portfolio.generation.validation import (
    validate_component_evidence,
    validate_project_evidence,
)
from genai_template.workflow.portfolio.ports.text_generator import TextGenerator


class SummaryGenerationResponse[SummaryT: StructuredSummary](_ImmutableDomainModel):
    """Parsed summary plus provider accounting and recovery warnings.

    Attributes:
        value:
            Typed summary parsed from the provider Markdown.
        provider:
            Provider that performed generation.
        model:
            Provider model that performed generation.
        token_usage:
            Complete, partial, or unavailable provider usage.
        provider_metadata:
            Narrow non-secret provider audit metadata.
        raw_response_hash:
            SHA-256 digest of the exact provider response text.
        warnings:
            Ordered deterministic Markdown-parser recoveries.
    """

    value: SummaryT = Field(description="Parsed and validated typed summary.")
    provider: str = Field(min_length=1, description="Generation provider identity.")
    model: str = Field(min_length=1, description="Generation model identity.")
    token_usage: TokenUsage = Field(description="Provider-reported token usage.")
    provider_metadata: ProviderAuditMetadata = Field(
        default_factory=ProviderAuditMetadata,
        description="Narrow non-secret provider audit metadata.",
    )
    raw_response_hash: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 digest of the exact provider response text.",
    )
    warnings: tuple[GenerationWarning, ...] = Field(
        default=(),
        description="Ordered deterministic Markdown-parser recoveries.",
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
