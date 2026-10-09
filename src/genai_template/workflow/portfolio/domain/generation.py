"""Immutable contracts for structured generation and generated artifacts."""

from __future__ import annotations

from pydantic import Field, JsonValue

from genai_template.workflow.portfolio.domain.artifacts import (
    ArtifactProvenance,
    GenerationWarning,
    ProviderAuditMetadata,
    TokenUsage,
)
from genai_template.workflow.portfolio.domain.base import ImmutableDomainModel
from genai_template.workflow.portfolio.domain.summaries import StructuredSummary


class GenerationRequest(ImmutableDomainModel):
    """One fully identified text-generation provider request.

    Attributes:
        provenance:
            Expected stable provenance and generation identity.
        prompt:
            Assembled prompt sent to the text-generation provider.
        input_paths:
            Ordered repository paths actually represented in the prompt.
    """

    provenance: ArtifactProvenance = Field(
        description="Expected provenance and generation identity."
    )
    prompt: str = Field(min_length=1, description="Assembled provider prompt.")
    input_paths: tuple[str, ...] = Field(
        description="Ordered repository paths represented in the prompt."
    )


class GenerationResult(ImmutableDomainModel):
    """Validated structured output and provider accounting for one request.

    Attributes:
        provenance:
            Stable provenance shared with the generation request.
        structured_output:
            JSON-compatible output validated against the requested schema.
        token_usage:
            Complete, partial, or unavailable provider token usage.
        provider_metadata:
            Narrow non-secret metadata retained for auditing.
    """

    provenance: ArtifactProvenance = Field(description="Stable artifact provenance.")
    structured_output: dict[str, JsonValue] = Field(
        description="JSON-compatible validated structured output."
    )
    token_usage: TokenUsage = Field(description="Reported provider token usage.")
    provider_metadata: ProviderAuditMetadata = Field(
        default_factory=ProviderAuditMetadata,
        description="Narrow non-secret provider audit metadata.",
    )


class SummaryGenerationResponse[SummaryT: StructuredSummary](ImmutableDomainModel):
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


class TextGenerationResponse(ImmutableDomainModel):
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
