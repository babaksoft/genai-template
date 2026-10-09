"""Tests for deterministic portfolio Markdown parsing and recovery."""

from __future__ import annotations

import pytest

from genai_template.workflow.portfolio.domain import (
    ArchitectureSummary,
    ComponentSummary,
    ProjectOverviewSummary,
)
from genai_template.workflow.portfolio.domain import (
    TestingOperationsSummary as OperationsSummary,
)
from genai_template.workflow.portfolio.infrastructure.generation import (
    ARCHITECTURE_SPECIFICATION,
    COMPONENT_SPECIFICATION,
    OVERVIEW_SPECIFICATION,
    SUMMARY_SPECIFICATIONS,
    TESTING_OPERATIONS_SPECIFICATION,
    SummarySpecification,
    parse_summary_markdown,
    validate_component_evidence,
    validate_project_evidence,
)


def _canonical_response(specification: SummarySpecification) -> str:
    """Build canonical Markdown for a summary specification.

    Args:
        specification:
            Ordered contract to render.

    Returns:
        Complete canonical provider Markdown.
    """

    return "\n\n".join(
        f"## {section.heading}\n\nObserved {section.heading.lower()}.\n\n"
        "### Evidence\n\n- src/a.py"
        for section in specification.sections
    )


@pytest.mark.parametrize(
    ("specification", "output_type"),
    [
        (COMPONENT_SPECIFICATION, ComponentSummary),
        (OVERVIEW_SPECIFICATION, ProjectOverviewSummary),
        (ARCHITECTURE_SPECIFICATION, ArchitectureSummary),
        (TESTING_OPERATIONS_SPECIFICATION, OperationsSummary),
    ],
)
def test_canonical_artifact_response_produces_typed_summary(
    specification: SummarySpecification,
    output_type: type[
        ComponentSummary
        | ProjectOverviewSummary
        | ArchitectureSummary
        | OperationsSummary
    ],
) -> None:
    """Every canonical artifact contract parses without recovery warnings.

    Args:
        specification:
            Artifact Markdown contract under test.
        output_type:
            Expected existing typed-summary model.
    """

    parsed = parse_summary_markdown(
        _canonical_response(specification), specification, ("src/a.py",)
    )

    assert isinstance(parsed.summary, output_type)
    assert parsed.warnings == ()


def test_parser_filters_mixed_evidence_and_keeps_every_valid_path() -> None:
    """Invalid citations are warned while all valid citations are retained."""

    response = _canonical_response(COMPONENT_SPECIFICATION).replace(
        "- src/a.py", "- src/z.py\n- invented.py\n- src/a.py", 1
    )

    parsed = parse_summary_markdown(
        response, COMPONENT_SPECIFICATION, ("src/z.py", "src/a.py")
    )

    assert isinstance(parsed.summary, ComponentSummary)
    assert parsed.summary.responsibilities.evidence_paths == (
        "src/a.py",
        "src/z.py",
    )
    assert [warning.code for warning in parsed.warnings] == ["invalid_evidence_path"]
    validate_component_evidence(parsed.summary, ("src/a.py", "src/z.py"))


@pytest.mark.parametrize(
    "evidence_block",
    ["- invented.py", ""],
)
def test_parser_falls_back_for_all_invalid_or_absent_evidence(
    evidence_block: str,
) -> None:
    """A section without a valid citation receives the exact allowed scope.

    Args:
        evidence_block:
            Invalid or absent evidence-list content.
    """

    response = _canonical_response(ARCHITECTURE_SPECIFICATION).replace(
        "- src/a.py", evidence_block, 1
    )

    parsed = parse_summary_markdown(
        response,
        ARCHITECTURE_SPECIFICATION,
        ("src/b.py", "src/a.py"),
    )

    assert isinstance(parsed.summary, ArchitectureSummary)
    assert parsed.summary.boundaries.evidence_paths == ("src/a.py", "src/b.py")
    codes = [warning.code for warning in parsed.warnings]
    if evidence_block:
        assert codes[:2] == [
            "invalid_evidence_path",
            "evidence_scope_fallback",
        ]
    else:
        assert codes[0] == "evidence_scope_fallback"
    validate_project_evidence(
        parsed.summary,
        repository_context_paths=("src/a.py",),
        component_evidence_paths=("src/b.py",),
    )


def test_parser_recovers_structure_and_preserves_unassigned_prose_stably() -> None:
    """Structural defects produce stable validated output and warning order."""

    response = """Preamble fact.

## Behavior

Runtime fact.

### Evidence

- src/a.py

## Responsibilities

### Evidence

- bad.py

## Behavior

Second runtime fact.

### Evidence

- src/a.py

## Surprise

Unexpected fact.

### Evidence

- src/a.py

Epilogue fact.
"""

    first = parse_summary_markdown(
        response, COMPONENT_SPECIFICATION, ("src/a.py", "tests/a.py")
    )
    second = parse_summary_markdown(
        response, COMPONENT_SPECIFICATION, ("tests/a.py", "src/a.py")
    )

    assert first == second
    assert isinstance(first.summary, ComponentSummary)
    assert first.summary.responsibilities.content == (
        "Preamble fact.\nUnexpected fact.\n- src/a.py\nEpilogue fact.",
    )
    assert first.summary.responsibilities.evidence_paths == (
        "src/a.py",
        "tests/a.py",
    )
    assert first.summary.important_abstractions.content == (
        "No supported information was provided.",
    )
    assert first.summary.constraints.content == (
        "No supported information was provided.",
    )
    assert [warning.code for warning in first.warnings] == [
        "reordered_section",
        "duplicate_section",
        "unexpected_section",
        "unassigned_content",
        "invalid_evidence_path",
        "evidence_scope_fallback",
        "missing_section",
        "evidence_scope_fallback",
        "missing_section",
        "evidence_scope_fallback",
        "missing_section",
        "evidence_scope_fallback",
    ]


def test_parser_replaces_empty_section_body_with_placeholder() -> None:
    """An explicitly empty section remains valid and inspectably recovered."""

    response = _canonical_response(OVERVIEW_SPECIFICATION).replace(
        "Observed purpose.\n\n", "", 1
    )

    parsed = parse_summary_markdown(response, OVERVIEW_SPECIFICATION, ("src/a.py",))

    assert isinstance(parsed.summary, ProjectOverviewSummary)
    assert parsed.summary.purpose.content == ("No supported information was provided.",)
    assert parsed.warnings[0].code == "missing_section"


def test_parser_rejects_empty_response_without_attempting_recovery() -> None:
    """An empty provider response remains a hard boundary failure."""

    with pytest.raises(ValueError, match="must be non-empty"):
        parse_summary_markdown(" \r\n", COMPONENT_SPECIFICATION, ("src/a.py",))


def test_specifications_and_exact_headings_are_immutable() -> None:
    """Contracts cannot be mutated and padded headings remain unexpected."""

    with pytest.raises(TypeError):
        SUMMARY_SPECIFICATIONS["component"] = OVERVIEW_SPECIFICATION  # type: ignore[index]

    response = _canonical_response(COMPONENT_SPECIFICATION).replace(
        "## Responsibilities", "## Responsibilities ", 1
    )
    parsed = parse_summary_markdown(response, COMPONENT_SPECIFICATION, ("src/a.py",))

    assert [warning.code for warning in parsed.warnings[:2]] == [
        "unexpected_section",
        "unassigned_content",
    ]
    assert any(warning.code == "missing_section" for warning in parsed.warnings)
