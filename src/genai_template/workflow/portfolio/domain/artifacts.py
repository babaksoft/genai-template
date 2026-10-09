from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import Field, JsonValue, PositiveInt, model_validator

from genai_template.workflow.portfolio.domain.base import ImmutableDomainModel
from genai_template.workflow.portfolio.domain.summaries import (
    ArchitectureSummary,
    ArtifactKind,
    ComponentSummary,
    GenerationWarning,
    GenerationWarningCode,
    ProjectArtifactKind,
    ProjectOverviewSummary,
    StructuredSummary,
    TestingOperationsSummary,
)


class TokenUsage(ImmutableDomainModel):
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


class ProviderAuditMetadata(ImmutableDomainModel):
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


class ArtifactProvenance(ImmutableDomainModel):
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


class CachedArtifact(ImmutableDomainModel):
    """Validated structured artifact suitable for persistent cache storage.

    Attributes:
        cache_schema_version:
            Version of the cache envelope schema.
        provenance:
            Stable non-secret generation provenance.
        summary:
            Validated JSON-compatible application-owned summary.
        output_hash:
            SHA-256 identity of the canonical validated summary.
        raw_response_hash:
            SHA-256 digest of the original provider response text.
        original_token_usage:
            Usage reported by the original provider call.
        original_estimated_cost:
            Optional original-call cost estimate.
        provider_metadata:
            Narrow non-secret provider audit metadata.
        generation_warnings:
            Ordered deterministic recoveries applied to the provider response.
    """

    cache_schema_version: str = Field(
        min_length=1,
        description="Version of the cache envelope schema.",
    )
    provenance: ArtifactProvenance = Field(description="Stable artifact provenance.")
    summary: dict[str, JsonValue] = Field(
        description="JSON-compatible validated application-owned summary."
    )
    output_hash: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 identity of the canonical validated summary.",
    )
    raw_response_hash: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 digest of the original provider response text.",
    )
    original_token_usage: TokenUsage = Field(
        description="Original provider token usage."
    )
    original_estimated_cost: Decimal | None = Field(
        default=None,
        ge=Decimal(0),
        description="Optional cost estimated for the original provider call.",
    )
    provider_metadata: ProviderAuditMetadata = Field(
        default_factory=ProviderAuditMetadata,
        description="Narrow non-secret provider audit metadata.",
    )
    generation_warnings: tuple[GenerationWarning, ...] = Field(
        default=(),
        description="Ordered deterministic provider-response recoveries.",
    )


class ProjectSummaryArtifact(ImmutableDomainModel):
    """One project summary together with its cache and run information.

    Attributes:
        artifact_kind:
            Balanced project-document type represented by the summary.
        summary:
            Validated structured project summary.
        artifact:
            Validated reusable cache artifact.
        run_report:
            Current-run cache, billing, and latency information.
    """

    artifact_kind: ProjectArtifactKind = Field(
        description="Balanced project-document type represented by the summary."
    )
    summary: StructuredSummary = Field(
        description="Validated structured project summary."
    )
    artifact: CachedArtifact = Field(description="Validated reusable cache artifact.")
    run_report: ArtifactRunReport = Field(
        description="Current-run cache, billing, and latency information."
    )

    @model_validator(mode="after")
    def validate_summary_kind(self) -> ProjectSummaryArtifact:
        """Require the artifact kind and concrete summary schema to agree.

        Returns:
            The validated project summary artifact.

        Raises:
            ValueError:
                If the artifact kind, summary, cache artifact, or report disagree.
        """

        expected_types: dict[ProjectArtifactKind, type[StructuredSummary]] = {
            "overview": ProjectOverviewSummary,
            "architecture": ArchitectureSummary,
            "testing_operations": TestingOperationsSummary,
        }
        if not isinstance(self.summary, expected_types[self.artifact_kind]):
            raise TypeError("project artifact kind does not match its summary schema")
        if self.artifact.provenance.artifact_kind != self.artifact_kind:
            raise ValueError("project artifact kind does not match its provenance")
        if self.run_report.artifact_kind != self.artifact_kind:
            raise ValueError("project artifact kind does not match its run report")
        return self


class ComponentSummaryArtifact(ImmutableDomainModel):
    """One component summary together with its cache and run information.

    Attributes:
        unit_id:
            Stable configured summary-unit identifier.
        summary:
            Validated structured component summary.
        artifact:
            Validated reusable cache artifact.
        run_report:
            Current-run cache, billing, and latency information.
    """

    unit_id: str = Field(
        min_length=1,
        description="Stable configured summary-unit identifier.",
    )
    summary: ComponentSummary = Field(
        description="Validated structured component summary."
    )
    artifact: CachedArtifact = Field(description="Validated reusable cache artifact.")
    run_report: ArtifactRunReport = Field(
        description="Current-run cache, billing, and latency information."
    )


class ArtifactRunReport(ImmutableDomainModel):
    """Volatile current-run metrics for one generated or cached artifact.

    Attributes:
        artifact_kind:
            Logical type of the artifact.
        generation_fingerprint:
            Stable generation identity of the artifact.
        cache_hit:
            Whether the artifact was reused without a provider call.
        token_usage:
            Usage billed during this run, unknown when unavailable.
        estimated_cost:
            Cost estimated from current-run billed usage.
        original_token_usage:
            Usage recorded by the provider call that created the artifact.
        original_estimated_cost:
            Cost recorded when the artifact was originally created.
        generation_warning_counts:
            Response-recovery counts keyed by stable warning code.
        latency_seconds:
            Current-run provider or cache lookup latency.
    """

    artifact_kind: ArtifactKind = Field(description="Logical generated-artifact type.")
    generation_fingerprint: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Stable generation identity of the artifact.",
    )
    cache_hit: bool = Field(description="Whether this run reused a cached artifact.")
    token_usage: TokenUsage = Field(description="Token usage billed during this run.")
    estimated_cost: Decimal | None = Field(
        default=None,
        ge=Decimal(0),
        description="Estimated cost billed during this run.",
    )
    original_token_usage: TokenUsage | None = Field(
        default=None,
        description="Original provider usage retained with the artifact.",
    )
    original_estimated_cost: Decimal | None = Field(
        default=None,
        ge=Decimal(0),
        description="Original estimated provider cost retained with the artifact.",
    )
    generation_warning_counts: dict[GenerationWarningCode, PositiveInt] = Field(
        default_factory=dict,
        description="Response-recovery counts keyed by stable warning code.",
    )
    latency_seconds: float = Field(
        ge=0.0,
        description="Current-run artifact latency in seconds.",
    )

    @property
    def generation_warning_count(self) -> int:
        """Return the total number of retained response recoveries.

        Returns:
            Sum of all per-code warning counts.
        """

        return sum(self.generation_warning_counts.values())
