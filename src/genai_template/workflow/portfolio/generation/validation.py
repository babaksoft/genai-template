"""Evidence-scope validation for generated structured summaries."""

from __future__ import annotations

from collections.abc import Iterable

from genai_template.workflow.portfolio.domain.summaries import (
    ComponentSummary,
    StructuredSummary,
    summary_evidence_paths,
)


class EvidenceValidationError(ValueError):
    """A generated summary cites evidence outside its exact input scope.

    Attributes:
        invalid_paths:
            Sorted paths that were not members of the permitted scope.
        permitted_paths:
            Sorted exact paths permitted for this generation call.
    """

    def __init__(
        self,
        message: str,
        *,
        invalid_paths: tuple[str, ...],
        permitted_paths: tuple[str, ...],
    ) -> None:
        """Initialize an evidence validation failure.

        Args:
            message:
                Human-readable failure description.
            invalid_paths:
                Paths cited outside the allowed scope.
            permitted_paths:
                Exact allowed evidence paths.
        """

        super().__init__(message)
        self.invalid_paths = invalid_paths
        self.permitted_paths = permitted_paths


def validate_component_evidence(
    summary: ComponentSummary,
    planned_paths: Iterable[str],
) -> ComponentSummary:
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


def validate_project_evidence(
    summary: StructuredSummary,
    repository_context_paths: Iterable[str],
    component_evidence_paths: Iterable[str],
) -> StructuredSummary:
    """Require project evidence to have been supplied to synthesis.

    Args:
        summary:
            Validated overview, architecture, or testing/operations output.
        repository_context_paths:
            Exact repository paths supplied directly to the synthesis call.
        component_evidence_paths:
            Evidence paths present in component artifacts supplied to the call.

    Returns:
        The unchanged validated summary.

    Raises:
        EvidenceValidationError:
            If any cited path was absent from both actual input scopes.
    """

    if isinstance(summary, ComponentSummary):
        raise TypeError("project evidence validation does not accept components")
    permitted_paths = set(repository_context_paths) | set(component_evidence_paths)
    _validate_scope(summary, permitted_paths)
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
