"""Port for provider-backed validated structured summary generation."""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import Field

from genai_template.workflow.portfolio.domain.generation import (
    GenerationRequest,
    ProviderAuditMetadata,
    TokenUsage,
)
from genai_template.workflow.portfolio.domain.snapshot import _ImmutableDomainModel
from genai_template.workflow.portfolio.domain.summaries import StructuredSummary

SummaryT = TypeVar("SummaryT", bound=StructuredSummary)


class StructuredGenerationResponse[SummaryT: StructuredSummary](_ImmutableDomainModel):
    """Validated provider output with non-secret accounting metadata.

    Attributes:
        value:
            Output validated against the requested Pydantic summary type.
        provider:
            Provider that performed the generation.
        model:
            Provider model that performed the generation.
        token_usage:
            Complete, partial, or unavailable token usage.
        provider_metadata:
            Narrow non-secret metadata retained for auditing.
    """

    value: SummaryT = Field(description="Validated structured summary value.")
    provider: str = Field(min_length=1, description="Generation provider identity.")
    model: str = Field(min_length=1, description="Generation model identity.")
    token_usage: TokenUsage = Field(description="Provider-reported token usage.")
    provider_metadata: ProviderAuditMetadata = Field(
        default_factory=ProviderAuditMetadata,
        description="Narrow non-secret provider audit metadata.",
    )


class StructuredSummaryGenerator(Protocol):
    """Provider-neutral interface for validated summary generation."""

    def generate(
        self,
        request: GenerationRequest,
        output_type: type[SummaryT],
    ) -> StructuredGenerationResponse[SummaryT]:
        """Generate and validate one structured summary.

        Args:
            request:
                Fully identified prompt request.
            output_type:
                Strict Pydantic output model expected for this artifact.

        Returns:
            Validated output plus provider accounting metadata.

        Raises:
            StructuredGenerationError:
                If the provider call or output validation fails.
        """

        ...
