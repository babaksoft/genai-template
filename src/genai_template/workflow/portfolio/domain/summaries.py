"""Strict structured summaries produced by the portfolio workflow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Literal

from pydantic import Field, field_validator

from genai_template.workflow.portfolio.domain.base import ImmutableDomainModel
from genai_template.workflow.portfolio.domain.snapshot import SnapshotFile

OUTPUT_SCHEMA_VERSION = "v1"

ArtifactKind = Literal["component", "overview", "architecture", "testing_operations"]
ProjectArtifactKind = Literal["overview", "architecture", "testing_operations"]
GenerationWarningCode = Literal[
    "missing_section",
    "duplicate_section",
    "reordered_section",
    "unexpected_section",
    "unassigned_content",
    "invalid_evidence_path",
    "evidence_scope_fallback",
]


class GenerationWarning(ImmutableDomainModel):
    """One typed deterministic recovery performed while parsing Markdown.

    Attributes:
        code:
            Stable machine-readable recovery category.
        section:
            Affected configured or unexpected heading when applicable.
        detail:
            Stable human-readable description without provider prose.
    """

    code: GenerationWarningCode = Field(description="Stable recovery category.")
    section: str | None = Field(
        default=None,
        description="Affected Markdown heading when applicable.",
    )
    detail: str = Field(
        min_length=1,
        description="Stable recovery description.",
    )


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


@dataclass(frozen=True)
class SummarySectionSpecification:
    """Map one exact Markdown heading to a structured-summary field.

    Attributes:
        heading:
            Exact level-two Markdown heading text without the marker.
        field_name:
            Existing structured-summary field populated from the section.
    """

    heading: str
    field_name: str


@dataclass(frozen=True)
class SummarySpecification:
    """Define the ordered Markdown contract for one artifact kind.

    Attributes:
        artifact_kind:
            Artifact kind governed by the contract.
        output_type:
            Existing Pydantic summary model produced by parsing.
        sections:
            Ordered exact heading-to-field mappings.
    """

    artifact_kind: ArtifactKind
    output_type: type[StructuredSummary]
    sections: tuple[SummarySectionSpecification, ...]

    def __post_init__(self) -> None:
        """Validate that the immutable mapping exactly covers its output model.

        Raises:
            ValueError:
                If headings or fields are duplicated, blank, or do not exactly
                cover the configured output model.
        """

        headings = tuple(section.heading for section in self.sections)
        fields = tuple(section.field_name for section in self.sections)
        if not self.sections or any(not heading for heading in headings):
            raise ValueError("summary specifications require non-empty headings")
        if len(headings) != len(set(headings)):
            raise ValueError("summary specification headings must be unique")
        if len(fields) != len(set(fields)):
            raise ValueError("summary specification fields must be unique")
        if fields != tuple(self.output_type.model_fields):
            raise ValueError(
                "summary specification fields must exactly match model field order"
            )


COMPONENT_SPECIFICATION = SummarySpecification(
    artifact_kind="component",
    output_type=ComponentSummary,
    sections=(
        SummarySectionSpecification("Responsibilities", "responsibilities"),
        SummarySectionSpecification("Important Abstractions", "important_abstractions"),
        SummarySectionSpecification("Behavior", "behavior"),
        SummarySectionSpecification("Constraints", "constraints"),
        SummarySectionSpecification("Testing Evidence", "testing_evidence"),
    ),
)
OVERVIEW_SPECIFICATION = SummarySpecification(
    artifact_kind="overview",
    output_type=ProjectOverviewSummary,
    sections=(
        SummarySectionSpecification("Purpose", "purpose"),
        SummarySectionSpecification("Capabilities", "capabilities"),
        SummarySectionSpecification("Entry Points", "entry_points"),
        SummarySectionSpecification("Technology Choices", "technology_choices"),
    ),
)
ARCHITECTURE_SPECIFICATION = SummarySpecification(
    artifact_kind="architecture",
    output_type=ArchitectureSummary,
    sections=(
        SummarySectionSpecification("Boundaries", "boundaries"),
        SummarySectionSpecification("Dependencies", "dependencies"),
        SummarySectionSpecification("Principal Flows", "principal_flows"),
    ),
)
TESTING_OPERATIONS_SPECIFICATION = SummarySpecification(
    artifact_kind="testing_operations",
    output_type=TestingOperationsSummary,
    sections=(
        SummarySectionSpecification("Testing Strategy", "testing_strategy"),
        SummarySectionSpecification("Local Operation", "local_operation"),
        SummarySectionSpecification("Configuration", "configuration"),
        SummarySectionSpecification("Observability", "observability"),
        SummarySectionSpecification(
            "Operational Constraints", "operational_constraints"
        ),
    ),
)

SUMMARY_SPECIFICATIONS: Mapping[ArtifactKind, SummarySpecification] = MappingProxyType(
    {
        specification.artifact_kind: specification
        for specification in (
            COMPONENT_SPECIFICATION,
            OVERVIEW_SPECIFICATION,
            ARCHITECTURE_SPECIFICATION,
            TESTING_OPERATIONS_SPECIFICATION,
        )
    }
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
