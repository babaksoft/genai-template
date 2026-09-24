"""Command-line inspection of immutable portfolio repository snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

from genai_template.workflow.portfolio.config import (
    ProjectConfig,
    load_portfolio_config,
)
from genai_template.workflow.portfolio.planning import build_summary_plan
from genai_template.workflow.portfolio.readers import (
    LocalGitSnapshotReader,
    RepositoryReader,
)
from genai_template.workflow.portfolio.selection import build_repository_snapshot


class SnapshotInspectionError(ValueError):
    """Failure to select or orchestrate a configured snapshot inspection."""


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
    """Serializable stable report for a portfolio inspection.

    Attributes:
        version:
            Inspection report schema version.
        projects:
            Inspected projects in configuration order.
    """

    version: int
    projects: tuple[ProjectInspection, ...]


def inspect_portfolio(
    config_path: Path,
    *,
    project_slug: str | None = None,
    reader: RepositoryReader | None = None,
) -> PortfolioInspection:
    """Inspect configured committed snapshots and planned summary work.

    Args:
        config_path:
            Portfolio YAML configuration path.
        project_slug:
            Optional slug restricting inspection to one project.
        reader:
            Optional provider-neutral reader used for repository access.

    Returns:
        Stable metadata report for the selected configured projects.

    Raises:
        SnapshotInspectionError:
            If the requested project slug is not configured.
        FileNotFoundError:
            If the configuration file does not exist.
        ValueError:
            If configuration, selection, or planning is invalid.
        RepositoryReadError:
            If a repository or ref cannot be read.
    """

    config = load_portfolio_config(config_path)
    projects = _select_projects(config.projects, project_slug)
    snapshot_reader = reader or LocalGitSnapshotReader()
    return PortfolioInspection(
        version=1,
        projects=tuple(
            _inspect_project(project, snapshot_reader) for project in projects
        ),
    )


def _select_projects(
    projects: tuple[ProjectConfig, ...], project_slug: str | None
) -> tuple[ProjectConfig, ...]:
    """Select configured projects without changing configuration order.

    Args:
        projects:
            Validated projects from the portfolio configuration.
        project_slug:
            Optional exact project slug filter.

    Returns:
        All configured projects or the single matching project.

    Raises:
        SnapshotInspectionError:
            If no project has the requested slug.
    """

    if project_slug is None:
        return projects
    selected = tuple(project for project in projects if project.slug == project_slug)
    if not selected:
        raise SnapshotInspectionError(
            f"Project slug is not configured: {project_slug!r}"
        )
    return selected


def _inspect_project(
    project: ProjectConfig, reader: RepositoryReader
) -> ProjectInspection:
    """Run the Stage 0 read, selection, and planning pipeline for one project.

    Args:
        project:
            Validated project configuration.
        reader:
            Provider-neutral immutable repository reader.

    Returns:
        Content-free inspection metadata for the project.
    """

    read_result = reader.read(project.repository)
    snapshot = build_repository_snapshot(
        project.slug,
        read_result,
        project.selection,
    )
    plan = build_summary_plan(snapshot, project.summary_units)
    return ProjectInspection(
        project_slug=project.slug,
        display_name=project.display_name,
        repository_url=snapshot.repository_url,
        requested_ref=snapshot.requested_ref,
        resolved_commit_sha=snapshot.resolved_commit_sha,
        selected_file_count=len(snapshot.files),
        total_bytes=sum(file.byte_size for file in snapshot.files),
        source_fingerprint=snapshot.source_fingerprint,
        summary_units=tuple(
            SummaryUnitInspection(
                unit_id=unit.unit_id,
                input_fingerprint=unit.input_fingerprint,
                member_paths=tuple(file.path for file in unit.files),
            )
            for unit in plan.units
        ),
    )


def render_text_report(report: PortfolioInspection) -> str:
    """Render an inspection report in a stable human-readable form.

    Args:
        report:
            Content-free portfolio inspection metadata.

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
            Content-free portfolio inspection metadata.

    Returns:
        Indented JSON with deterministic key ordering and no trailing newline.
    """

    return json.dumps(asdict(report), indent=2, sort_keys=True)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Parser for snapshot inspection command arguments.
    """

    parser = argparse.ArgumentParser(
        description="Inspect immutable portfolio repository snapshots."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="portfolio YAML configuration path",
    )
    parser.add_argument(
        "--project",
        "--project-slug",
        dest="project_slug",
        help="inspect only the project with this slug",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="emit the stable JSON report",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the snapshot inspection command.

    Args:
        argv:
            Optional command arguments excluding the executable name.
        stdout:
            Optional standard-output replacement for callers and tests.
        stderr:
            Optional standard-error replacement for callers and tests.

    Returns:
        Zero after a successful report, otherwise one for inspection failures.
    """

    output = stdout or sys.stdout
    error_output = stderr or sys.stderr
    arguments = _build_argument_parser().parse_args(argv)
    try:
        report = inspect_portfolio(
            arguments.config,
            project_slug=arguments.project_slug,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=error_output)
        return 1

    rendered = (
        render_json_report(report)
        if arguments.json_output
        else render_text_report(report)
    )
    print(rendered, file=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
