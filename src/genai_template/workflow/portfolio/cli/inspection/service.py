"""Application service for inspecting immutable Portfolio snapshots."""

from __future__ import annotations

from pathlib import Path

from genai_template.workflow.portfolio.adapters.repositories.local_git import (
    LocalGitSnapshotReader,
)
from genai_template.workflow.portfolio.cli.inspection.reports import (
    PortfolioInspection,
    ProjectInspection,
    SummaryUnitInspection,
)
from genai_template.workflow.portfolio.config import (
    ProjectConfig,
    load_portfolio_config,
)
from genai_template.workflow.portfolio.ports.repository import RepositoryReader
from genai_template.workflow.portfolio.snapshot.planning import build_summary_plan
from genai_template.workflow.portfolio.snapshot.selection import (
    build_repository_snapshot,
)


class SnapshotInspectionError(ValueError):
    """Failure to select or orchestrate a configured snapshot inspection."""


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
            Validated projects from the Portfolio configuration.
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
