"""Stable report models and renderers for Portfolio snapshot inspection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SummaryUnitInspection:
    """Serializable inspection metadata for one planned summary unit.

    Attributes:
        unit_id:
            Stable configured summary-unit identifier.
        input_fingerprint:
            SHA-256 identity of the unit's inputs.
        member_paths:
            Ordered repository-relative paths assigned to the unit.
    """

    unit_id: str
    input_fingerprint: str
    member_paths: tuple[str, ...]


@dataclass(frozen=True)
class ProjectInspection:
    """Serializable inspection metadata for one configured project.

    The report deliberately excludes repository-local paths and source contents.

    Attributes:
        project_slug:
            Stable configured project identifier.
        display_name:
            Human-readable configured project name.
        repository_url:
            Normalized repository URL when available.
        requested_ref:
            Mutable ref requested by configuration.
        resolved_commit_sha:
            Immutable commit selected by the reader.
        selected_file_count:
            Number of normalized files in the snapshot.
        total_bytes:
            Combined normalized UTF-8 size of the selected files.
        source_fingerprint:
            SHA-256 identity of selection settings and contents.
        summary_units:
            Ordered logical units planned from the snapshot.
        warnings:
            Non-fatal conditions tolerated during inspection.
    """

    project_slug: str
    display_name: str
    repository_url: str | None
    requested_ref: str
    resolved_commit_sha: str
    selected_file_count: int
    total_bytes: int
    source_fingerprint: str
    summary_units: tuple[SummaryUnitInspection, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PortfolioInspection:
    """Serializable stable report for a Portfolio inspection.

    Attributes:
        version:
            Inspection report schema version.
        projects:
            Inspected projects in configuration order.
    """

    version: int
    projects: tuple[ProjectInspection, ...]


def render_text_report(report: PortfolioInspection) -> str:
    """Render an inspection report in a stable human-readable form.

    Args:
        report:
            Content-free Portfolio inspection metadata.

    Returns:
        Deterministically formatted multiline text without a trailing newline.
    """

    lines: list[str] = []
    for project_index, project in enumerate(report.projects):
        if project_index:
            lines.append("")
        lines.extend(
            [
                f"Project: {project.project_slug} ({project.display_name})",
                f"Requested ref: {project.requested_ref}",
                f"Resolved commit: {project.resolved_commit_sha}",
            ]
        )
        if project.repository_url is not None:
            lines.append(f"Repository URL: {project.repository_url}")
        lines.extend(
            [
                f"Selected files: {project.selected_file_count}",
                f"Total bytes: {project.total_bytes}",
                f"Source fingerprint: {project.source_fingerprint}",
                "Summary units:",
            ]
        )
        for unit in project.summary_units:
            lines.append(f"  - {unit.unit_id}: {unit.input_fingerprint}")
            lines.extend(f"    - {path}" for path in unit.member_paths)
        if project.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {warning}" for warning in project.warnings)
    return "\n".join(lines)


def render_json_report(report: PortfolioInspection) -> str:
    """Render an inspection report as stable, content-free JSON.

    Args:
        report:
            Content-free Portfolio inspection metadata.

    Returns:
        Indented JSON with deterministic key ordering and no trailing newline.
    """

    return json.dumps(asdict(report), indent=2, sort_keys=True)
