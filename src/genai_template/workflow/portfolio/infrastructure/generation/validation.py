"""Evidence-scope validation for generated structured summaries."""

from __future__ import annotations

from collections.abc import Iterable

from genai_template.workflow.portfolio.domain import (
    ComponentSummary,
    EvidenceValidationError,
    StructuredSummary,
    summary_evidence_paths,
)


def validate_evidence(
    summary: StructuredSummary,
    planned_paths: Iterable[str],
) -> StructuredSummary:
    """Require component evidence to belong to that component's planned files.

    Args:
        summary:
            Validated component structured output.
        planned_paths:
            Exact file paths assigned to the component.

    Returns:
        The unchanged validated summary.

    Raises:
        EvidenceValidationError:
            If any cited path is outside the component input scope.
    """

    _validate_scope(summary, planned_paths)
    return summary


def _validate_scope(summary: StructuredSummary, permitted: Iterable[str]) -> None:
    """Validate exact membership of all cited paths.

    Args:
        summary:
            Structured summary to validate.
        permitted:
            Exact evidence scope for the provider call.

    Raises:
        EvidenceValidationError:
            If a cited path is outside the allowed scope.
    """

    permitted_paths = tuple(sorted(set(permitted)))
    permitted_set = set(permitted_paths)
    invalid_paths = tuple(
        path for path in summary_evidence_paths(summary) if path not in permitted_set
    )
    if invalid_paths:
        raise EvidenceValidationError(
            "generated evidence is outside the exact generation input scope",
            invalid_paths=invalid_paths,
            permitted_paths=permitted_paths,
        )
