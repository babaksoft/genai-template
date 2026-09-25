"""Immutable execution-report contracts for portfolio corpus generation."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from pydantic import Field

from genai_template.workflow.portfolio.domain.generation import (
    ArtifactKind,
    TokenUsage,
)
from genai_template.workflow.portfolio.domain.snapshot import _ImmutableDomainModel


class ArtifactRunReport(_ImmutableDomainModel):
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
    latency_seconds: float = Field(
        ge=0.0,
        description="Current-run artifact latency in seconds.",
    )


class GenerationRunReport(_ImmutableDomainModel):
    """Volatile report for one complete portfolio corpus run.

    Attributes:
        project_slug:
            Stable project identifier.
        resolved_commit_sha:
            Immutable commit processed by the run.
        source_fingerprint:
            Stable selected-source identity.
        corpus_fingerprint:
            Stable published corpus identity when publication succeeds.
        artifacts:
            Ordered per-artifact execution reports.
        provider_call_count:
            Number of billable provider calls made by this run.
        billed_token_usage:
            Aggregate current-run token usage.
        estimated_cost:
            Aggregate current-run cost estimate when available.
        elapsed_seconds:
            End-to-end current-run latency.
        published_path:
            Resolved published corpus pointer when publication succeeds.
        release_path:
            Resolved immutable release path when publication succeeds.
    """

    project_slug: str = Field(min_length=1, description="Stable project identifier.")
    resolved_commit_sha: str = Field(
        pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$",
        description="Immutable commit processed by the run.",
    )
    source_fingerprint: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Stable selected-source identity.",
    )
    corpus_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description="Stable corpus identity after successful publication.",
    )
    artifacts: tuple[ArtifactRunReport, ...] = Field(
        description="Ordered per-artifact execution reports."
    )
    provider_call_count: int = Field(
        ge=0,
        description="Billable provider calls made by this run.",
    )
    billed_token_usage: TokenUsage = Field(
        description="Aggregate current-run provider token usage."
    )
    estimated_cost: Decimal | None = Field(
        default=None,
        ge=Decimal(0),
        description="Aggregate current-run estimated cost.",
    )
    elapsed_seconds: float = Field(
        ge=0.0,
        description="End-to-end run latency in seconds.",
    )
    published_path: Path | None = Field(
        default=None,
        description="Resolved published corpus pointer after success.",
    )
    release_path: Path | None = Field(
        default=None,
        description="Resolved immutable release path after success.",
    )
