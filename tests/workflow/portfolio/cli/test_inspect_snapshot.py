"""Tests for Stage 0 Portfolio snapshot-inspection behavior."""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from genai_template.workflow.portfolio.cli.inspect import main
from genai_template.workflow.portfolio.cli.inspection import (
    SnapshotInspectionError,
    inspect_portfolio,
    render_json_report,
    render_text_report,
)


def _git(repository_path: Path, *arguments: str) -> str:
    """Run Git in a temporary test repository.

    Args:
        repository_path:
            Repository in which to run Git.
        *arguments:
            Git subcommand and arguments.

    Returns:
        Decoded standard output from the successful command.
    """

    result = subprocess.run(
        ("git", "-C", str(repository_path), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_project_config(
    config_path: Path,
    repository_path: Path,
    *,
    slug: str = "sample-project",
    display_name: str = "Sample Project",
) -> None:
    """Write a complete inspection configuration for a test repository.

    Args:
        config_path:
            YAML destination path.
        repository_path:
            Absolute local Git repository path.
        slug:
            Configured project slug.
        display_name:
            Configured project display name.
    """

    data = {
        "version": 1,
        "projects": [
            {
                "slug": slug,
                "display_name": display_name,
                "repository": {
                    "type": "local_git",
                    "path": str(repository_path),
                    "ref": "HEAD",
                    "url": "https://example.test/owner/sample.git",
                },
                "selection": {
                    "include": ["README.md", "src/**/*.py"],
                    "exclude": ["src/excluded.py"],
                    "max_file_bytes": 1024,
                },
                "summary_units": [
                    {
                        "id": "application",
                        "paths": ["README.md", "src/**/*.py"],
                    }
                ],
            }
        ],
    }
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


@pytest.fixture
def inspection_fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    """Create a committed repository and matching portfolio configuration.

    Args:
        tmp_path:
            Pytest temporary directory.

    Returns:
        Configuration path, repository path, and resolved commit SHA.
    """

    repository_path = tmp_path / "private-local-repository"
    repository_path.mkdir()
    _git(repository_path, "init", "--initial-branch=main")
    _git(repository_path, "config", "user.name", "Test User")
    _git(repository_path, "config", "user.email", "test@example.test")
    (repository_path / "src").mkdir()
    (repository_path / "README.md").write_text("secret readme\n", encoding="utf-8")
    (repository_path / "src" / "app.py").write_text(
        "secret = 'do not report'\n", encoding="utf-8"
    )
    (repository_path / "src" / "excluded.py").write_bytes(b"ignored\0content")
    _git(repository_path, "add", ".")
    _git(repository_path, "commit", "-m", "fixture")
    commit_sha = _git(repository_path, "rev-parse", "HEAD")
    config_path = tmp_path / "portfolio.yml"
    _write_project_config(config_path, repository_path)
    return config_path, repository_path, commit_sha


def test_text_and_json_reports_are_stable_and_content_free(
    inspection_fixture: tuple[Path, Path, str],
) -> None:
    """Both formats expose identities and paths without local paths or contents."""

    config_path, repository_path, commit_sha = inspection_fixture

    first = inspect_portfolio(config_path)
    second = inspect_portfolio(config_path)
    text_report = render_text_report(first)
    json_report = render_json_report(first)

    assert first == second
    assert render_text_report(second) == text_report
    assert render_json_report(second) == json_report
    assert f"Resolved commit: {commit_sha}" in text_report
    assert "Project: sample-project (Sample Project)" in text_report
    assert "Repository URL: https://example.test/owner/sample" in text_report
    assert "Selected files: 2" in text_report
    assert "Total bytes: 39" in text_report
    assert "README.md" in text_report
    assert "src/app.py" in text_report
    assert str(repository_path) not in text_report
    assert str(repository_path) not in json_report
    assert "secret readme" not in text_report
    assert "secret readme" not in json_report
    assert "do not report" not in text_report
    assert "do not report" not in json_report

    parsed = json.loads(json_report)
    assert parsed["version"] == 1
    assert parsed["projects"][0]["resolved_commit_sha"] == commit_sha
    assert parsed["projects"][0]["warnings"] == []
    assert parsed["projects"][0]["summary_units"][0]["member_paths"] == [
        "README.md",
        "src/app.py",
    ]


def test_dirty_work_tree_does_not_change_report(
    inspection_fixture: tuple[Path, Path, str],
) -> None:
    """Uncommitted, staged, and untracked changes cannot affect inspection."""

    config_path, repository_path, _ = inspection_fixture
    original = render_json_report(inspect_portfolio(config_path))
    (repository_path / "src" / "app.py").write_text("staged change\n", encoding="utf-8")
    _git(repository_path, "add", "src/app.py")
    (repository_path / "src" / "app.py").write_text(
        "working tree change\n", encoding="utf-8"
    )
    (repository_path / "untracked.py").write_text("untracked\n", encoding="utf-8")

    changed = render_json_report(inspect_portfolio(config_path))

    assert changed == original


def test_project_filter_selects_one_configured_project(
    inspection_fixture: tuple[Path, Path, str], tmp_path: Path
) -> None:
    """The optional slug filter inspects only its exact configured project."""

    config_path, repository_path, _ = inspection_fixture
    first_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    second_path = tmp_path / "second.yml"
    _write_project_config(
        second_path,
        repository_path,
        slug="second-project",
        display_name="Second Project",
    )
    second_data = yaml.safe_load(second_path.read_text(encoding="utf-8"))
    first_data["projects"].extend(second_data["projects"])
    config_path.write_text(
        yaml.safe_dump(first_data, sort_keys=False), encoding="utf-8"
    )

    report = inspect_portfolio(config_path, project_slug="second-project")

    assert [project.project_slug for project in report.projects] == ["second-project"]


def test_missing_project_filter_fails_before_repository_access(
    inspection_fixture: tuple[Path, Path, str],
) -> None:
    """An unknown project slug produces a focused orchestration failure."""

    config_path, _, _ = inspection_fixture

    with pytest.raises(SnapshotInspectionError, match="missing-project"):
        inspect_portfolio(config_path, project_slug="missing-project")


@pytest.mark.parametrize("failure", ["configuration", "repository", "planning"])
def test_cli_returns_nonzero_for_stage_failures(
    inspection_fixture: tuple[Path, Path, str], tmp_path: Path, failure: str
) -> None:
    """Configuration, repository, and planning failures return exit status one.

    Args:
        inspection_fixture:
            Committed repository and configuration fixture.
        tmp_path:
            Pytest temporary directory.
        failure:
            Stage boundary to make invalid.
    """

    config_path, _, _ = inspection_fixture
    if failure == "configuration":
        config_path.write_text("projects: [", encoding="utf-8")
    else:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if failure == "repository":
            data["projects"][0]["repository"]["path"] = str(tmp_path / "missing")
        else:
            data["projects"][0]["summary_units"][0]["paths"] = ["tests/**/*.py"]
        config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()

    status = main(["--config", str(config_path)], stdout=stdout, stderr=stderr)

    assert status == 1
    assert stdout.getvalue() == ""
    assert stderr.getvalue().startswith("error: ")


def test_cli_json_mode_and_project_alias(
    inspection_fixture: tuple[Path, Path, str],
) -> None:
    """The CLI writes valid JSON and accepts the explicit project-slug option."""

    config_path, _, _ = inspection_fixture
    stdout = io.StringIO()
    stderr = io.StringIO()

    status = main(
        [
            "--config",
            str(config_path),
            "--project-slug",
            "sample-project",
            "--json",
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert status == 0
    assert stderr.getvalue() == ""
    assert json.loads(stdout.getvalue())["projects"][0]["project_slug"] == (
        "sample-project"
    )
