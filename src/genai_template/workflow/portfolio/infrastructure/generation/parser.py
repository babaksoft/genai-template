"""Deterministic parser for provider-generated portfolio Markdown."""

from __future__ import annotations

from collections.abc import Iterable

from genai_template.workflow.portfolio.domain import (
    EvidenceSection,
    GenerationWarning,
    ParsedSummary,
    SummarySpecification,
    SummarySectionSpecification,
)
from genai_template.workflow.portfolio.infrastructure.generation.validation import (
    validate_evidence,
)

_PLACEHOLDER = "No supported information was provided."


class HeadingParser:
    def __init__(
        self,
        specification: SummarySpecification,
        allowed_evidence_paths: Iterable[str]
    ) -> None:
        if not allowed_evidence_paths:
            raise ValueError("allowed evidence scope must be non-empty")

        self._allowed_paths = tuple(sorted(set(allowed_evidence_paths)))
        self._specification = specification
        self._headings = {
            section.heading: section for section in specification.sections
        }
        self._content: dict[str, list[str]] = {
            section.field_name: [] for section in specification.sections
        }
        self._evidence: dict[str, list[str]] = {
            section.field_name: [] for section in specification.sections
        }
        self._seen: set[str] = set()
        self._warnings: list[GenerationWarning] = []
        self._section_indexes = {
            section.heading: index for index, section in enumerate(specification.sections)
        }
        self._current_field: str | None = None
        self._in_evidence = False
        self._unassigned: list[str] = []
        self._blocks: tuple[str, ...] = ()

    def examine_heading(self, line: str) -> None:
        heading = line[3:]
        section = self._headings.get(heading)
        if section is None:
            self._warnings.append(
                GenerationWarning(
                    code="unexpected_section",
                    section=heading or None,
                    detail="Unexpected Markdown section was reassigned.",
                )
            )
            self._current_field = None
            return

        self._current_field = section.field_name
        section_index = self._section_indexes[heading]
        if heading in self._seen:
            self._warnings.append(
                GenerationWarning(
                    code="duplicate_section",
                    section=heading,
                    detail="Duplicate section content was merged.",
                )
            )
        else:
            if section_index < highest_section_index:
                self._warnings.append(
                    GenerationWarning(
                        code="reordered_section",
                        section=heading,
                        detail="Out-of-order section was mapped by heading.",
                    )
                )
            self._seen.add(heading)
            highest_section_index = max(highest_section_index, section_index)

    def parse_line(self, line: str) -> None:
        if line == "### Evidence":
            self._in_evidence = self._current_field is not None
            return

        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            self._warnings.append(
                GenerationWarning(
                    code="unexpected_section",
                    section=heading or None,
                    detail="Unexpected Markdown heading was reassigned.",
                )
            )
            self._current_field = None
            self._in_evidence = False
            return

        if self._current_field is None:
            self._unassigned.append(line)
        elif self._in_evidence:
            if not line.strip():
                return

            if line.startswith("- ") and line[2:].strip():
                self._evidence[self._current_field].append(line[2:].strip())
            else:
                self._unassigned.append(line)
        else:
            self._content[self._current_field].append(line)

    def handle_unassigned(self) -> None:
        unassigned_blocks = _content_blocks(self._unassigned)
        if unassigned_blocks:
            primary = self._specification.sections[0]
            self._content[primary.field_name].extend(unassigned_blocks)
            self._warnings.append(
                GenerationWarning(
                    code="unassigned_content",
                    section=primary.heading,
                    detail="Unassigned prose was preserved in the primary section.",
                )
            )

    def examine_section(self, section: SummarySectionSpecification) -> None:
        self._blocks = _content_blocks(self._content[section.field_name])
        if section.heading not in self._seen:
            self._warnings.append(
                GenerationWarning(
                    code="missing_section",
                    section=section.heading,
                    detail="Missing section was populated deterministically.",
                )
            )

        if not self._blocks:
            if section.heading in self._seen:
                self._warnings.append(
                    GenerationWarning(
                        code="missing_section",
                        section=section.heading,
                        detail="Empty section was populated deterministically.",
                    )
                )
            self._blocks = (_PLACEHOLDER,)

    def parse_section(
        self, section: SummarySectionSpecification
    ) -> EvidenceSection:
        valid_paths: set[str] = set()
        invalid_paths: set[str] = set()
        for path in self._evidence[section.field_name]:
            if path in self._allowed_paths:
                valid_paths.add(path)
            else:
                invalid_paths.add(path)

        for path in sorted(invalid_paths):
            self._warnings.append(
                GenerationWarning(
                    code="invalid_evidence_path",
                    section=section.heading,
                    detail=f"Discarded out-of-scope evidence path: {path}",
                )
            )

        if not valid_paths:
            valid_paths.update(self._allowed_paths)
            self._warnings.append(
                GenerationWarning(
                    code="evidence_scope_fallback",
                    section=section.heading,
                    detail="Allowed call scope replaced absent valid evidence.",
                )
            )

        parsed_section = EvidenceSection(
            content=self._blocks,
            evidence_paths=tuple(sorted(valid_paths)),
        )
        self._blocks = ()

        return parsed_section


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

    warnings: list[GenerationWarning] = []

    parser = HeadingParser(specification, allowed_evidence_paths)
    lines = normalized.split("\n")
    for line in lines:
        if line.startswith("## ") and not line.startswith("### "):
            parser.examine_heading(line)
        parser.parse_line(line)

    parser.handle_unassigned()

    output: dict[str, EvidenceSection] = {}
    for section in specification.sections:
        parser.examine_section(section)
        output[section.field_name] = parser.parse_section(section)

    summary = specification.output_type.model_validate(output)
    allowed_paths = tuple(sorted(set(allowed_evidence_paths)))
    validate_evidence(summary, allowed_paths)

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
