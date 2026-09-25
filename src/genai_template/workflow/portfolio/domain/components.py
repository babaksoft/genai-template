"""Domain result returned by cached component-summary generation."""

from pydantic import Field

from genai_template.workflow.portfolio.domain.generation import CachedArtifact
from genai_template.workflow.portfolio.domain.reports import ArtifactRunReport
from genai_template.workflow.portfolio.domain.snapshot import _ImmutableDomainModel
from genai_template.workflow.portfolio.domain.summaries import ComponentSummary


class ComponentSummaryArtifact(_ImmutableDomainModel):
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
