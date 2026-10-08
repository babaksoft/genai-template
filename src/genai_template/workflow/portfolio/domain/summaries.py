"""Strict structured summaries produced by the portfolio workflow."""

from __future__ import annotations

from pathlib import PurePosixPath

from pydantic import Field, field_validator

from genai_template.workflow.portfolio.domain.base import ImmutableDomainModel
from genai_template.workflow.portfolio.domain.generation import GenerationWarning
from genai_template.workflow.portfolio.domain.snapshot import SnapshotFile

OUTPUT_SCHEMA_VERSION = "v1"


class SummaryUnitPlan(ImmutableDomainModel):
    """The exact snapshot files used to produce one logical summary.

    Attributes:
        unit_id:
            Stable identifier of the configured logical summary unit.
        input_fingerprint:
            SHA-256 identity of this unit's settings and selected contents.
        files:
            Non-empty ordered snapshot files assigned to the unit.
    """

    unit_id: str = Field(
        min_length=1,
        description="Stable logical summary-unit identifier.",
    )
    input_fingerprint: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 identity of the unit settings and contents.",
    )
    files: tuple[SnapshotFile, ...] = Field(
        min_length=1,
        description="Ordered snapshot files assigned to this summary unit.",
    )


class SummaryPlan(ImmutableDomainModel):
    """Stable logical summary work derived from a repository snapshot.

    Attributes:
        project_slug:
            Stable project identifier shared with the source snapshot.
        resolved_commit_sha:
            Full object identifier of the snapshot's resolved commit.
        source_fingerprint:
            SHA-256 identity shared with the source snapshot.
        units:
            Non-empty ordered logical summary units to process.
    """

    project_slug: str = Field(
        min_length=1,
        description="Stable project identifier shared with the snapshot.",
    )
    resolved_commit_sha: str = Field(
        pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$",
        description="Full hexadecimal identifier of the resolved commit.",
    )
    source_fingerprint: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 identity shared with the source snapshot.",
    )
    units: tuple[SummaryUnitPlan, ...] = Field(
        min_length=1,
        description="Ordered logical summary units to process.",
    )


class ParsedSummary(ImmutableDomainModel):
    """Validated summary and ordered parser recoveries.

    Attributes:
        summary:
            Existing validated typed summary produced from Markdown.
        warnings:
            Recoveries in deterministic parser order.
    """

    summary: StructuredSummary = Field(description="Validated parsed summary.")
    warnings: tuple[GenerationWarning, ...] = Field(
        description="Ordered deterministic parser recoveries."
    )


class EvidenceSection(ImmutableDomainModel):
    """One summary section whose claims share explicit source evidence.

    Attributes:
        content:
            Concise factual statements for this section.
        evidence_paths:
            Sorted repository paths supporting the section.
    """

    content: tuple[str, ...] = Field(
        min_length=1,
        description="Concise factual statements for this section.",
    )
    evidence_paths: tuple[str, ...] = Field(
        min_length=1,
        description="Sorted repository-relative paths supporting the section.",
    )

    @field_validator("content")
    @classmethod
    def validate_content(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Reject empty or padded statements.

        Args:
            values:
                Section statements supplied by a provider.

        Returns:
            The validated statements.

        Raises:
            ValueError:
                If a statement is blank or padded.
        """

        if any(not value or value != value.strip() for value in values):
            raise ValueError("section content must be non-empty and normalized")
        return values

    @field_validator("evidence_paths")
    @classmethod
    def validate_evidence_paths(cls, paths: tuple[str, ...]) -> tuple[str, ...]:
        """Validate evidence path representation.

        Args:
            paths:
                Evidence paths supplied by a provider.

        Returns:
            The validated paths.
        """

        return _validate_evidence_paths(paths)


class ComponentSummary(ImmutableDomainModel):
    """Validated summary of one configured logical component.

    Attributes:
        responsibilities:
            The component's primary responsibilities.
        important_abstractions:
            Important types, interfaces, and modules.
        behavior:
            Significant runtime behavior and interactions.
        constraints:
            Design limitations, assumptions, and invariants.
        testing_evidence:
            Tests and verification relevant to the component.
    """

    responsibilities: EvidenceSection = Field(
        description="The component's primary responsibilities and evidence."
    )
    important_abstractions: EvidenceSection = Field(
        description="Important component abstractions and evidence."
    )
    behavior: EvidenceSection = Field(
        description="Significant component behavior and evidence."
    )
    constraints: EvidenceSection = Field(
        description="Component constraints, assumptions, and evidence."
    )
    testing_evidence: EvidenceSection = Field(
        description="Component tests and verification evidence."
    )


class ArchitectureSummary(ImmutableDomainModel):
    """Validated project architecture summary.

    Attributes:
        boundaries:
            Major system boundaries and their roles.
        dependencies:
            Significant internal and external dependencies.
        principal_flows:
            Principal data and control flows.
    """

    boundaries: EvidenceSection = Field(
        description="Major architecture boundaries and evidence."
    )
    dependencies: EvidenceSection = Field(
        description="Significant architecture dependencies and evidence."
    )
    principal_flows: EvidenceSection = Field(
        description="Principal architecture flows and evidence."
    )


class ProjectOverviewSummary(ImmutableDomainModel):
    """Validated high-level project overview.

    Attributes:
        purpose:
            The problem and audience served by the project.
        capabilities:
            User-visible and system capabilities.
        entry_points:
            Important ways to run or interact with the project.
        technology_choices:
            Material technologies and their observed roles.
    """

    purpose: EvidenceSection = Field(description="Project purpose and evidence.")
    capabilities: EvidenceSection = Field(
        description="Project capabilities and evidence."
    )
    entry_points: EvidenceSection = Field(
        description="Project entry points and evidence."
    )
    technology_choices: EvidenceSection = Field(
        description="Material technology choices and evidence."
    )


class TestingOperationsSummary(ImmutableDomainModel):
    """Validated project testing and operations summary.

    Attributes:
        testing_strategy:
            Test layers, tools, and important test boundaries.
        local_operation:
            How the project is run in a local environment.
        configuration:
            Operational configuration and environment inputs.
        observability:
            Logging, metrics, tracing, and health facilities.
        operational_constraints:
            Known operational limitations and external requirements.
    """

    testing_strategy: EvidenceSection = Field(
        description="Testing strategy and evidence."
    )
    local_operation: EvidenceSection = Field(
        description="Local operation guidance and evidence."
    )
    configuration: EvidenceSection = Field(
        description="Operational configuration and evidence."
    )
    observability: EvidenceSection = Field(
        description="Observability facilities and evidence."
    )
    operational_constraints: EvidenceSection = Field(
        description="Operational constraints and evidence."
    )


StructuredSummary = (
    ComponentSummary
    | ArchitectureSummary
    | ProjectOverviewSummary
    | TestingOperationsSummary
)


def summary_evidence_paths(summary: StructuredSummary) -> tuple[str, ...]:
    """Collect deterministic unique evidence paths from a structured summary.

    Args:
        summary:
            Validated structured summary.

    Returns:
        Sorted unique evidence paths used by any section.
    """

    paths: set[str] = set()
    for field_name in type(summary).model_fields:
        section = getattr(summary, field_name)
        paths.update(section.evidence_paths)
    return tuple(sorted(paths))


def _validate_evidence_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    """Validate canonical evidence paths.

    Args:
        paths:
            Repository-relative evidence paths supplied by a provider.

    Returns:
        The validated paths.

    Raises:
        ValueError:
            If paths are blank, unsafe, duplicated, or not sorted canonically.
    """

    for path in paths:
        if not path or path != path.strip():
            raise ValueError("evidence paths must be non-empty and normalized")
        if "\\" in path or path.startswith("/"):
            raise ValueError("evidence paths must be repository-relative POSIX paths")
        parts = path.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError("evidence paths must be normalized")
        if PurePosixPath(path).is_absolute():
            raise ValueError("evidence paths must be repository-relative")

    if len(paths) != len(set(paths)):
        raise ValueError("evidence paths must be deduplicated")
    if paths != tuple(sorted(paths)):
        raise ValueError("evidence paths must be deterministically sorted")

    return paths
