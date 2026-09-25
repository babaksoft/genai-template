"""Immutable contracts for structured generation and generated artifacts."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from genai_template.workflow.portfolio.domain.snapshot import _ImmutableDomainModel

ArtifactKind = Literal["component", "overview", "architecture", "testing_operations"]


class TokenUsage(_ImmutableDomainModel):
    """Provider token counts, preserving unavailable counts as unknown.

    Attributes:
        input_tokens:
            Reported input-token count, or null when unavailable.
        output_tokens:
            Reported output-token count, or null when unavailable.
    """

    input_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Reported input-token count, or null when unavailable.",
    )
    output_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Reported output-token count, or null when unavailable.",
    )


class ProviderAuditMetadata(_ImmutableDomainModel):
    """Non-secret provider metadata retained for artifact auditing.

    Attributes:
        request_id:
            Optional provider request identifier.
        finish_reason:
            Optional provider completion reason.
    """

    request_id: str | None = Field(
        default=None,
        description="Optional provider request identifier.",
    )
    finish_reason: str | None = Field(
        default=None,
        description="Optional provider completion reason.",
    )


class ArtifactProvenance(_ImmutableDomainModel):
    """Stable, non-secret provenance for one generated artifact.

    Attributes:
        artifact_kind:
            Logical type of generated artifact.
        generation_fingerprint:
            SHA-256 identity of every content-affecting generation input.
        source_fingerprint:
            SHA-256 identity of the selected immutable repository snapshot.
        unit_input_fingerprint:
            Component-unit identity when generating a component artifact.
        component_artifact_hashes:
            Ordered validated component hashes consumed by project synthesis.
        prompt_id:
            Stable prompt-template identifier.
        prompt_hash:
            SHA-256 hash of the exact prompt template.
        output_schema_version:
            Version of the validated structured output.
        provider:
            Structured-generation provider identifier.
        model:
            Provider model name.
        temperature:
            Sampling temperature affecting generated content.
        seed:
            Optional sampling seed affecting generated content.
    """

    artifact_kind: ArtifactKind = Field(description="Logical generated-artifact type.")
    generation_fingerprint: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 identity of content-affecting generation inputs.",
    )
    source_fingerprint: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 identity of the immutable source snapshot.",
    )
    unit_input_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description="Component-unit input identity when applicable.",
    )
    component_artifact_hashes: tuple[str, ...] = Field(
        default=(),
        description="Ordered component artifact hashes consumed by synthesis.",
    )
    prompt_id: str = Field(min_length=1, description="Stable prompt identifier.")
    prompt_hash: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 hash of the prompt template.",
    )
    output_schema_version: str = Field(
        min_length=1,
        description="Validated structured-output schema version.",
    )
    provider: Literal["ollama", "openai"] = Field(
        description="Structured-generation provider identifier."
    )
    model: str = Field(min_length=1, description="Provider model name.")
    temperature: float = Field(
        ge=0.0,
        le=2.0,
        description="Content-affecting sampling temperature.",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        description="Optional content-affecting sampling seed.",
    )

    @model_validator(mode="after")
    def validate_dependencies(self) -> ArtifactProvenance:
        """Keep component and project dependency identities unambiguous.

        Returns:
            The validated artifact provenance.

        Raises:
            ValueError:
                If dependency fields do not match the artifact kind.
        """

        if self.artifact_kind == "component":
            if self.unit_input_fingerprint is None:
                raise ValueError("component provenance requires a unit fingerprint")
            if self.component_artifact_hashes:
                raise ValueError("component provenance cannot depend on components")
        elif self.unit_input_fingerprint is not None:
            raise ValueError("project provenance cannot contain a unit fingerprint")
        return self


class GenerationRequest(_ImmutableDomainModel):
    """One fully identified structured-generation provider request.

    Attributes:
        provenance:
            Expected stable provenance and generation identity.
        prompt:
            Assembled prompt sent to the structured-generation provider.
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


class GenerationResult(_ImmutableDomainModel):
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


class CachedArtifact(_ImmutableDomainModel):
    """Validated structured artifact suitable for persistent cache storage.

    Attributes:
        cache_schema_version:
            Version of the cache envelope schema.
        provenance:
            Stable non-secret generation provenance.
        structured_output:
            Validated JSON-compatible structured output.
        output_hash:
            SHA-256 identity of the canonical structured output.
        token_usage:
            Usage reported by the original provider call.
        estimated_cost:
            Optional original-call cost estimate.
        provider_metadata:
            Narrow non-secret provider audit metadata.
    """

    cache_schema_version: str = Field(
        min_length=1,
        description="Version of the cache envelope schema.",
    )
    provenance: ArtifactProvenance = Field(description="Stable artifact provenance.")
    structured_output: dict[str, JsonValue] = Field(
        description="JSON-compatible validated structured output."
    )
    output_hash: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 identity of the canonical structured output.",
    )
    token_usage: TokenUsage = Field(description="Original provider token usage.")
    estimated_cost: Decimal | None = Field(
        default=None,
        ge=Decimal(0),
        description="Optional cost estimated for the original provider call.",
    )
    provider_metadata: ProviderAuditMetadata = Field(
        default_factory=ProviderAuditMetadata,
        description="Narrow non-secret provider audit metadata.",
    )


class RenderedDocument(_ImmutableDomainModel):
    """One deterministic Markdown document derived from an artifact.

    Attributes:
        filename:
            Safe flat corpus filename.
        document_type:
            Logical type of the rendered document.
        component_id:
            Component unit identifier for component documents.
        generation_fingerprint:
            Generation identity of the source artifact.
        artifact_hash:
            SHA-256 identity of the structured source artifact.
        content:
            Deterministically rendered Markdown text.
        content_hash:
            SHA-256 identity of the UTF-8 Markdown bytes.
        evidence_paths:
            Ordered repository evidence paths represented by the document.
    """

    filename: str = Field(min_length=1, description="Safe flat corpus filename.")
    document_type: ArtifactKind = Field(description="Logical document type.")
    component_id: str | None = Field(
        default=None,
        description="Component unit identifier when applicable.",
    )
    generation_fingerprint: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Generation identity of the source artifact.",
    )
    artifact_hash: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 identity of the structured source artifact.",
    )
    content: str = Field(description="Deterministically rendered Markdown text.")
    content_hash: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 identity of the UTF-8 Markdown bytes.",
    )
    evidence_paths: tuple[str, ...] = Field(
        description="Ordered repository evidence paths."
    )


class CorpusBuild(_ImmutableDomainModel):
    """Complete in-memory document set awaiting corpus publication.

    Attributes:
        project_slug:
            Stable project identifier.
        source_fingerprint:
            SHA-256 identity of the immutable source snapshot.
        resolved_commit_sha:
            Immutable repository commit used for the build.
        documents:
            Ordered rendered Markdown documents in the build.
    """

    project_slug: str = Field(min_length=1, description="Stable project identifier.")
    source_fingerprint: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 identity of the immutable source snapshot.",
    )
    resolved_commit_sha: str = Field(
        pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$",
        description="Immutable repository commit used for the build.",
    )
    documents: tuple[RenderedDocument, ...] = Field(
        min_length=1,
        description="Ordered rendered Markdown documents in the build.",
    )
