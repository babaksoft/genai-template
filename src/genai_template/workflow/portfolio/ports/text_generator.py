"""Port for provider-backed plain-text generation."""

from __future__ import annotations

from typing import Protocol

from pydantic import Field

from genai_template.workflow.portfolio.domain.generation import (
    GenerationRequest,
    ProviderAuditMetadata,
    TokenUsage,
)
from genai_template.workflow.portfolio.domain.snapshot import _ImmutableDomainModel


class TextGenerationResponse(_ImmutableDomainModel):
    """Plain provider output with non-secret accounting metadata.

    Attributes:
        text:
            Complete non-empty provider response text.
        provider:
            Provider that performed the generation.
        model:
            Provider model that performed the generation.
        token_usage:
            Complete, partial, or unavailable token usage.
        provider_metadata:
            Narrow non-secret metadata retained for auditing.
    """

    text: str = Field(min_length=1, description="Complete provider response text.")
    provider: str = Field(min_length=1, description="Generation provider identity.")
    model: str = Field(min_length=1, description="Generation model identity.")
    token_usage: TokenUsage = Field(description="Provider-reported token usage.")
    provider_metadata: ProviderAuditMetadata = Field(
        default_factory=ProviderAuditMetadata,
        description="Narrow non-secret provider audit metadata.",
    )


class TextGenerator(Protocol):
    """Provider-neutral interface for one-call plain-text generation."""

    def generate(self, request: GenerationRequest) -> TextGenerationResponse:
        """Generate one complete plain-text response.

        Args:
            request:
                Fully identified prompt request.

        Returns:
            Plain text plus provider identity and accounting metadata.

        Raises:
            TextGenerationError:
                If request identity, transport, or completion validation fails.
        """

        ...
