"""Tests for exact generated-evidence scope validation."""

from __future__ import annotations

import pytest

from genai_template.workflow.portfolio.domain import (
    ArchitectureSummary,
    ComponentSummary,
    EvidenceSection,
)
from genai_template.workflow.portfolio.generation.validation import (
    EvidenceValidationError,
    validate_component_evidence,
    validate_project_evidence,
)


def _section(path: str) -> EvidenceSection:
    """Build a one-path evidence section.

    Args:
        path:
            Evidence path.

    Returns:
        Evidence section.
    """

    return EvidenceSection(content=("Fact.",), evidence_paths=(path,))


def _component(path: str) -> ComponentSummary:
    """Build a component summary citing one path.

    Args:
        path:
            Evidence path used by all sections.

    Returns:
        Component summary.
    """

    section = _section(path)
    return ComponentSummary(
        responsibilities=section,
        important_abstractions=section,
        behavior=section,
        constraints=section,
        testing_evidence=section,
    )


def test_component_evidence_requires_exact_planned_file_membership() -> None:
    """A path from another component is rejected even if in the snapshot."""

    with pytest.raises(EvidenceValidationError) as caught:
        validate_component_evidence(_component("src/other.py"), ("src/mine.py",))

    assert caught.value.invalid_paths == ("src/other.py",)


def test_project_evidence_accepts_context_or_component_evidence() -> None:
    """Project documents can cite only the two input scopes they consumed."""

    summary = ArchitectureSummary(
        boundaries=_section("README.md"),
        dependencies=_section("src/component.py"),
        principal_flows=_section("README.md"),
    )

    assert (
        validate_project_evidence(
            summary,
            repository_context_paths=("README.md",),
            component_evidence_paths=("src/component.py",),
        )
        is summary
    )


def test_project_evidence_rejects_snapshot_path_not_supplied_to_call() -> None:
    """Snapshot membership alone does not make a path valid synthesis evidence."""

    summary = ArchitectureSummary(
        boundaries=_section("README.md"),
        dependencies=_section("src/unseen.py"),
        principal_flows=_section("README.md"),
    )

    with pytest.raises(EvidenceValidationError, match="exact generation input"):
        validate_project_evidence(summary, ("README.md",), ("src/supplied.py",))
