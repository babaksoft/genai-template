"""Domain results returned by cached project-summary synthesis."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from genai_template.workflow.portfolio.domain.generation import CachedArtifact
from genai_template.workflow.portfolio.domain.reports import ArtifactRunReport
from genai_template.workflow.portfolio.domain.snapshot import _ImmutableDomainModel
from genai_template.workflow.portfolio.domain.summaries import (
    ArchitectureSummary,
    ProjectOverviewSummary,
    StructuredSummary,
    TestingOperationsSummary,
)

ProjectArtifactKind = Literal["overview", "architecture", "testing_operations"]


class ProjectSummaryArtifact(_ImmutableDomainModel):
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
