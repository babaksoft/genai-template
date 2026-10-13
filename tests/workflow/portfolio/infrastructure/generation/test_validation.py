"""Tests for exact generated-evidence scope validation."""

from __future__ import annotations

import pytest

from genai_template.workflow.portfolio.domain import (
    ArchitectureSummary,
    ComponentSummary,
    EvidenceSection,
)
from genai_template.workflow.portfolio.domain.errors import EvidenceValidationError
from genai_template.workflow.portfolio.infrastructure.generation import (
    validate_evidence,
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


def test_evidence_requires_exact_planned_file_membership() -> None:
    """A path from another component is rejected even if in the snapshot."""

    with pytest.raises(EvidenceValidationError) as caught:
        validate_evidence(_component("src/other.py"), ("src/mine.py",))

    assert caught.value.invalid_paths == ("src/other.py",)
