"""Tests for portfolio configuration and Stage 0 domain models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from genai_template.config import settings
from genai_template.workflow.portfolio import (
    CommittedRepositoryEntry,
    LocalGitRepositoryConfig,
    PortfolioConfig,
    ProjectConfig,
    RepositorySnapshot,
    SelectionConfig,
    SnapshotFile,
    SummaryPlan,
    SummaryUnitConfig,
    SummaryUnitPlan,
    load_portfolio_config,
)


def _valid_config() -> dict[str, Any]:
    """Build a complete valid portfolio configuration mapping.

    Returns:
        Independent mutable configuration data for one project.
    """

    return {
        "version": 1,
        "projects": [
            {
                "slug": "sample-project",
                "display_name": "Sample Project",
                "repository": {"type": "local_git", "path": ".", "ref": "HEAD"},
                "selection": {
                    "include": ["README.md", "src/**/*.py"],
                    "exclude": ["**/__pycache__/**"],
                    "max_file_bytes": 1024,
                },
                "summary_units": [{"id": "application-core", "paths": ["src/**/*.py"]}],
            }
        ],
    }


def _write_config(tmp_path: Path, data: object) -> Path:
    """Write YAML configuration data into a temporary file.

    Args:
        tmp_path:
            Pytest temporary directory.
        data:
            YAML-compatible value to serialize.

    Returns:
        Path to the created YAML file.
    """

    config_path = tmp_path / "portfolio.yml"
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return config_path


def test_loads_documented_configuration() -> None:
    """The packaged one-project example is complete and valid."""

    config_path = settings.PKG_ROOT / "workflow" / "configs" / "portfolio-local.yml"

    config = load_portfolio_config(config_path)

    assert config.version == 1
    assert len(config.projects) == 1
    project = config.projects[0]
    assert project.slug == "genai-template"
    assert isinstance(project.repository, LocalGitRepositoryConfig)
    assert project.repository.path == settings.REPO_ROOT
    assert project.repository.ref == "HEAD"
    assert project.selection.max_file_bytes == 262144
    assert project.summary_units[0].id == "components-splitters"


def test_relative_config_and_repository_paths_use_repository_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither relative path depends on the process working directory."""

    relative_config_path = Path(
        "src/genai_template/workflow/configs/portfolio-local.yml"
    )
    monkeypatch.chdir(tmp_path)

    config = load_portfolio_config(relative_config_path)

    assert config.projects[0].repository.path == settings.REPO_ROOT


def test_absolute_repository_path_is_preserved(tmp_path: Path) -> None:
    """An explicit absolute local repository path remains absolute."""

    data = _valid_config()
    data["projects"][0]["repository"]["path"] = str(tmp_path)

    config = load_portfolio_config(_write_config(tmp_path, data))

    assert config.projects[0].repository.path == tmp_path.resolve()


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data.update(version=2),
        lambda data: data.update(projects=[]),
        lambda data: data.update(unexpected=True),
        lambda data: data["projects"][0].update(unexpected=True),
        lambda data: data["projects"][0]["repository"].pop("type"),
        lambda data: data["projects"][0]["repository"].update(type="github"),
        lambda data: data["projects"][0]["selection"].update(include=[]),
        lambda data: data["projects"][0].update(summary_units=[]),
        lambda data: data["projects"][0]["selection"].update(max_file_bytes=0),
        lambda data: data["projects"][0].update(slug="Not Normalized"),
        lambda data: data["projects"][0]["repository"].update(ref=" "),
    ],
    ids=[
        "unsupported-version",
        "empty-projects",
        "unknown-root-key",
        "unknown-project-key",
        "missing-source-type",
        "unsupported-source",
        "empty-includes",
        "empty-units",
        "invalid-size",
        "invalid-slug",
        "empty-ref",
    ],
)
def test_invalid_configuration_is_rejected(tmp_path: Path, change: Any) -> None:
    """Schema and boundary violations fail during configuration loading."""

    data = _valid_config()
    change(data)

    with pytest.raises(ValidationError):
        load_portfolio_config(_write_config(tmp_path, data))


def test_duplicate_project_slugs_are_rejected(tmp_path: Path) -> None:
    """A project identity cannot occur twice in one configuration."""

    data = _valid_config()
    data["projects"].append(data["projects"][0].copy())

    with pytest.raises(ValidationError, match="project slugs must be unique"):
        load_portfolio_config(_write_config(tmp_path, data))


def test_duplicate_unit_ids_are_rejected(tmp_path: Path) -> None:
    """Unit identifiers must be unique within their project."""

    data = _valid_config()
    unit = data["projects"][0]["summary_units"][0]
    data["projects"][0]["summary_units"].append(unit.copy())

    with pytest.raises(ValidationError, match="summary unit ids must be unique"):
        load_portfolio_config(_write_config(tmp_path, data))


@pytest.mark.parametrize(
    "pattern",
    [
        "/etc/passwd",
        "../secrets.txt",
        "src/../secrets.txt",
        "src//module.py",
        "src/./module.py",
        r"src\module.py",
        r"C:\repository\file.py",
        " src/**/*.py",
        "",
    ],
)
def test_unsafe_or_unnormalized_patterns_are_rejected(
    tmp_path: Path, pattern: str
) -> None:
    """File patterns must be normalized paths confined to a repository."""

    data = _valid_config()
    data["projects"][0]["selection"]["include"] = [pattern]

    with pytest.raises(ValidationError):
        load_portfolio_config(_write_config(tmp_path, data))


@pytest.mark.parametrize("contents", ["projects: [", "- project\n", "", "null\n"])
def test_malformed_or_non_mapping_yaml_is_rejected(
    tmp_path: Path, contents: str
) -> None:
    """Malformed YAML and non-mapping document roots fail clearly."""

    config_path = tmp_path / "invalid.yml"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError):
        load_portfolio_config(config_path)


def test_missing_configuration_file_is_rejected(tmp_path: Path) -> None:
    """A missing configuration retains the standard file-not-found signal."""

    with pytest.raises(FileNotFoundError):
        load_portfolio_config(tmp_path / "missing.yml")


def test_configuration_models_are_immutable(tmp_path: Path) -> None:
    """Loaded configuration values cannot be modified after validation."""

    config = load_portfolio_config(_write_config(tmp_path, _valid_config()))

    with pytest.raises(ValidationError):
        config.projects[0].slug = "changed"


def test_domain_models_are_immutable_and_composable() -> None:
    """Canonical Stage 0 domain values compose without repository access."""

    content_hash = "1" * 64
    commit_sha = "2" * 40
    source_fingerprint = "3" * 64
    unit_fingerprint = "4" * 64
    file = SnapshotFile(
        path="src/example.py",
        text="print('example')\n",
        content_hash=content_hash,
        byte_size=17,
    )
    snapshot = RepositorySnapshot(
        project_slug="sample-project",
        repository_url=None,
        requested_ref="HEAD",
        resolved_commit_sha=commit_sha,
        source_fingerprint=source_fingerprint,
        files=(file,),
    )
    plan = SummaryPlan(
        project_slug=snapshot.project_slug,
        resolved_commit_sha=snapshot.resolved_commit_sha,
        source_fingerprint=snapshot.source_fingerprint,
        units=(
            SummaryUnitPlan(
                unit_id="application-core",
                input_fingerprint=unit_fingerprint,
                files=snapshot.files,
            ),
        ),
    )
    entry = CommittedRepositoryEntry(
        path="src/example.py",
        mode="100644",
        object_type="blob",
        object_id=commit_sha,
        content=b"print('example')\n",
    )

    assert plan.units[0].files == snapshot.files
    assert entry.content == b"print('example')\n"
    with pytest.raises(ValidationError):
        file.text = "changed"


def test_unknown_domain_model_fields_are_rejected() -> None:
    """Workflow boundary values reject accidental additional fields."""

    with pytest.raises(ValidationError):
        PortfolioConfig.model_validate(
            {"version": 1, "projects": [], "unexpected": True}
        )


@pytest.mark.parametrize(
    "model_type",
    [
        LocalGitRepositoryConfig,
        SelectionConfig,
        SummaryUnitConfig,
        ProjectConfig,
        PortfolioConfig,
        CommittedRepositoryEntry,
        SnapshotFile,
        RepositorySnapshot,
        SummaryUnitPlan,
        SummaryPlan,
    ],
)
def test_public_models_document_classes_and_fields(
    model_type: type[BaseModel],
) -> None:
    """Every public Pydantic model documents its attributes and schema fields."""

    assert model_type.__doc__ is not None
    assert "Attributes:" in model_type.__doc__
    assert all(field.description for field in model_type.model_fields.values())
