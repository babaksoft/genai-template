"""Immutable Markdown contracts for portfolio summary artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from genai_template.workflow.portfolio.domain.artifacts import ArtifactKind
from genai_template.workflow.portfolio.domain.summaries import (
    ArchitectureSummary,
    ComponentSummary,
    ProjectOverviewSummary,
    StructuredSummary,
    TestingOperationsSummary,
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
