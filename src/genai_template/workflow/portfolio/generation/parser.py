"""Deterministic parser for provider-generated portfolio Markdown."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import Field

from genai_template.workflow.portfolio.domain.snapshot import _ImmutableDomainModel
from genai_template.workflow.portfolio.domain.summaries import (
    ComponentSummary,
    EvidenceSection,
    StructuredSummary,
)
from genai_template.workflow.portfolio.generation.specifications import (
    SummarySpecification,
)
from genai_template.workflow.portfolio.generation.validation import (
    validate_component_evidence,
    validate_project_evidence,
)

_PLACEHOLDER = "No supported information was provided."

GenerationWarningCode = Literal[
    "missing_section",
    "duplicate_section",
    "reordered_section",
    "unexpected_section",
    "unassigned_content",
    "invalid_evidence_path",
    "evidence_scope_fallback",
]


class GenerationWarning(_ImmutableDomainModel):
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
        description="Stable recovery description without provider prose.",
    )


class ParsedSummary(_ImmutableDomainModel):
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


def parse_summary_markdown(
    response_text: str,
    specification: SummarySpecification,
    allowed_evidence_paths: Iterable[str],
) -> ParsedSummary:
    """Parse provider Markdown into an existing typed summary model.

    Structural defects are recovered without another provider call. Evidence is
    filtered by exact membership in the supplied call scope, and the complete
    scope is used when a section retains no valid provider citation.

    Args:
        response_text:
            Non-empty provider response containing Markdown.
        specification:
            Exact ordered section contract for the requested artifact.
        allowed_evidence_paths:
            Exact repository-relative evidence scope supplied to the call.

    Returns:
        Validated typed summary and ordered recovery warnings.

    Raises:
        ValueError:
            If the response or allowed evidence scope is empty.
    """

    normalized = response_text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.strip():
        raise ValueError("provider Markdown response must be non-empty")

    allowed_paths = tuple(sorted(set(allowed_evidence_paths)))
    if not allowed_paths:
        raise ValueError("allowed evidence scope must be non-empty")

    headings = {section.heading: section for section in specification.sections}
    content: dict[str, list[str]] = {
        section.field_name: [] for section in specification.sections
    }
    evidence: dict[str, list[str]] = {
        section.field_name: [] for section in specification.sections
    }
    seen: set[str] = set()
    warnings: list[GenerationWarning] = []
    unassigned: list[str] = []
    current_field: str | None = None
    in_evidence = False
    highest_section_index = -1
    section_indexes = {
        section.heading: index for index, section in enumerate(specification.sections)
    }

    for line in normalized.split("\n"):
        if line.startswith("## ") and not line.startswith("### "):
            heading = line[3:]
            section = headings.get(heading)
            in_evidence = False
            if section is None:
                warnings.append(
                    GenerationWarning(
                        code="unexpected_section",
                        section=heading or None,
                        detail="Unexpected Markdown section was reassigned.",
                    )
                )
                current_field = None
                continue

            current_field = section.field_name
            section_index = section_indexes[heading]
            if heading in seen:
                warnings.append(
                    GenerationWarning(
                        code="duplicate_section",
                        section=heading,
                        detail="Duplicate section content was merged.",
                    )
                )
            else:
                if section_index < highest_section_index:
                    warnings.append(
                        GenerationWarning(
                            code="reordered_section",
                            section=heading,
                            detail="Out-of-order section was mapped by heading.",
                        )
                    )
                seen.add(heading)
                highest_section_index = max(highest_section_index, section_index)
            continue

        if line == "### Evidence":
            in_evidence = current_field is not None
            continue

        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            warnings.append(
                GenerationWarning(
                    code="unexpected_section",
                    section=heading or None,
                    detail="Unexpected Markdown heading was reassigned.",
                )
            )
            current_field = None
            in_evidence = False
            continue

        if current_field is None:
            unassigned.append(line)
        elif in_evidence:
            if not line.strip():
                continue
            if line.startswith("- ") and line[2:].strip():
                evidence[current_field].append(line[2:].strip())
            else:
                unassigned.append(line)
        else:
            content[current_field].append(line)

    unassigned_blocks = _content_blocks(unassigned)
    if unassigned_blocks:
        primary = specification.sections[0]
        content[primary.field_name].extend(unassigned_blocks)
        warnings.append(
            GenerationWarning(
                code="unassigned_content",
                section=primary.heading,
                detail="Unassigned prose was preserved in the primary section.",
            )
        )

    allowed_set = set(allowed_paths)
    output: dict[str, EvidenceSection] = {}
    for section in specification.sections:
        blocks = _content_blocks(content[section.field_name])
        if section.heading not in seen:
            warnings.append(
                GenerationWarning(
                    code="missing_section",
                    section=section.heading,
                    detail="Missing section was populated deterministically.",
                )
            )
        if not blocks:
            if section.heading in seen:
                warnings.append(
                    GenerationWarning(
                        code="missing_section",
                        section=section.heading,
                        detail="Empty section was populated deterministically.",
                    )
                )
            blocks = (_PLACEHOLDER,)

        valid_paths: set[str] = set()
        invalid_paths: set[str] = set()
        for path in evidence[section.field_name]:
            if path in allowed_set:
                valid_paths.add(path)
            else:
                invalid_paths.add(path)
        for path in sorted(invalid_paths):
            warnings.append(
                GenerationWarning(
                    code="invalid_evidence_path",
                    section=section.heading,
                    detail=f"Discarded out-of-scope evidence path: {path}",
                )
            )
        if not valid_paths:
            valid_paths.update(allowed_paths)
            warnings.append(
                GenerationWarning(
                    code="evidence_scope_fallback",
                    section=section.heading,
                    detail="Allowed call scope replaced absent valid evidence.",
                )
            )

        output[section.field_name] = EvidenceSection(
            content=blocks,
            evidence_paths=tuple(sorted(valid_paths)),
        )

    summary = specification.output_type.model_validate(output)
    if isinstance(summary, ComponentSummary):
        validate_component_evidence(summary, allowed_paths)
    else:
        validate_project_evidence(summary, allowed_paths, ())
    return ParsedSummary(summary=summary, warnings=tuple(warnings))


def _content_blocks(lines: Iterable[str]) -> tuple[str, ...]:
    """Normalize lines into non-empty prose blocks separated by blank lines.

    Args:
        lines:
            Provider lines assigned to one content destination.

    Returns:
        Normalized non-empty prose blocks in input order.
    """

    blocks: list[str] = []
    current: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            current.append(stripped)
        elif current:
            blocks.append("\n".join(current))
            current = []
    if current:
        blocks.append("\n".join(current))
    return tuple(blocks)
