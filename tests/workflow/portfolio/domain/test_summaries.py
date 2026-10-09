"""Tests for strict structured summary output models."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from genai_template.workflow.portfolio.domain import (
    ArchitectureSummary,
    ComponentSummary,
    EvidenceSection,
    ProjectOverviewSummary,
)
from genai_template.workflow.portfolio.domain import (
    TestingOperationsSummary as OperationsSummary,
)
from genai_template.workflow.portfolio.domain import summary_evidence_paths


def _section(*paths: str) -> EvidenceSection:
    """Build a valid evidence section.

    Args:
        paths:
            Sorted repository paths.

    Returns:
        Valid evidence section.
    """

    return EvidenceSection(content=("Observed behavior.",), evidence_paths=paths)


def test_summary_models_are_strict_immutable_and_documented() -> None:
    """Every structured output model exposes documented immutable fields."""

    models: tuple[type[BaseModel], ...] = (
        EvidenceSection,
        ComponentSummary,
        ArchitectureSummary,
        ProjectOverviewSummary,
        OperationsSummary,
    )

    for model in models:
        assert model.__doc__ is not None
        assert "Attributes:" in model.__doc__
        assert all(field.description for field in model.model_fields.values())

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EvidenceSection.model_validate(
            {
                "content": ("Fact.",),
                "evidence_paths": ("src/a.py",),
                "unexpected": True,
            }
        )


@pytest.mark.parametrize(
    "paths, message",
    [
        (("",), "non-empty"),
        (("../secret",), "normalized"),
        (("src\\a.py",), "POSIX"),
        (("src/a.py", "src/a.py"), "deduplicated"),
        (("src/z.py", "src/a.py"), "sorted"),
    ],
)
def test_evidence_paths_must_be_canonical(paths: tuple[str, ...], message: str) -> None:
    """Invalid evidence representation is rejected before scope validation.

    Args:
        paths:
            Invalid evidence paths.
        message:
            Expected validation error fragment.
    """

    with pytest.raises(ValidationError, match=message):
        _section(*paths)


def test_component_summary_collects_sorted_unique_evidence() -> None:
    """Evidence collection is deterministic across structured sections."""

    summary = ComponentSummary(
        responsibilities=_section("src/a.py"),
        important_abstractions=_section("src/b.py"),
        behavior=_section("src/a.py", "src/b.py"),
        constraints=_section("src/a.py"),
        testing_evidence=_section("tests/test_a.py"),
    )

    assert summary_evidence_paths(summary) == (
        "src/a.py",
        "src/b.py",
        "tests/test_a.py",
    )
