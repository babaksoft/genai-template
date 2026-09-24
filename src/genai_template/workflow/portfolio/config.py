"""Validated configuration loading for the portfolio snapshot workflow."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from genai_template.config import settings

_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_WINDOWS_ABSOLUTE_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")


def _validate_identifier(value: str, field_name: str) -> str:
    """Validate a normalized, lowercase kebab-case identifier.

    Args:
        value:
            Identifier supplied by configuration.
        field_name:
            Human-readable field name for validation errors.

    Returns:
        The validated identifier.

    Raises:
        ValueError:
            If the value is not normalized lowercase kebab case.
    """

    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be lowercase kebab case")
    return value


def _validate_ref(value: str) -> str:
    """Validate a non-empty repository ref without silently normalizing it.

    Args:
        value:
            Git ref supplied by configuration.

    Returns:
        The validated ref.

    Raises:
        ValueError:
            If the ref is empty, padded, or contains control characters.
    """

    if not value or value != value.strip():
        raise ValueError("repository ref must be non-empty and normalized")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("repository ref must not contain control characters")
    return value


def _validate_repository_pattern(value: str) -> str:
    """Validate a normalized repository-relative POSIX glob pattern.

    Args:
        value:
            Pattern supplied by configuration.

    Returns:
        The validated pattern.

    Raises:
        ValueError:
            If the pattern is empty, absolute, traversing, or not normalized.
    """

    if not value or value != value.strip():
        raise ValueError("repository pattern must be non-empty and normalized")
    if "\\" in value:
        raise ValueError("repository pattern must use POSIX separators")
    if value.startswith("/") or _WINDOWS_ABSOLUTE_PATTERN.match(value):
        raise ValueError("repository pattern must be relative")

    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(
            "repository pattern must be normalized and must not contain '..'"
        )
    if PurePosixPath(value).is_absolute():
        raise ValueError("repository pattern must be relative")
    return value


class _ImmutableConfigModel(BaseModel):
    """Base model for strict, immutable portfolio configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class LocalGitRepositoryConfig(_ImmutableConfigModel):
    """Settings for reading a commit from a local Git repository.

    Attributes:
        type:
            Discriminator identifying the local Git repository provider.
        path:
            Absolute path to the local Git repository.
        ref:
            Git reference to resolve to an immutable commit.
        url:
            Optional canonical repository URL overriding the Git origin.
    """

    type: Literal["local_git"] = Field(description="Repository provider discriminator.")
    path: Path = Field(description="Absolute path to the local Git repository.")
    ref: str = Field(
        default="HEAD",
        description="Git reference to resolve to an immutable commit.",
    )
    url: str | None = Field(
        default=None,
        min_length=1,
        description="Optional canonical repository URL overriding the Git origin.",
    )

    @field_validator("ref")
    @classmethod
    def validate_ref(cls, value: str) -> str:
        """Validate the configured Git ref.

        Args:
            value:
                Git ref supplied by configuration.

        Returns:
            The validated ref.
        """

        return _validate_ref(value)


RepositoryConfig = Annotated[
    LocalGitRepositoryConfig,
    Field(discriminator="type"),
]


class SelectionConfig(_ImmutableConfigModel):
    """Allowlist-first repository file-selection settings.

    Attributes:
        include:
            Repository-relative patterns defining candidate files.
        exclude:
            Repository-relative patterns removed from the included candidates.
        max_file_bytes:
            Maximum accepted size of an individual committed file.
    """

    include: tuple[str, ...] = Field(
        min_length=1,
        description="Repository-relative patterns defining candidate files.",
    )
    exclude: tuple[str, ...] = Field(
        default=(),
        description="Patterns removed after include patterns are applied.",
    )
    max_file_bytes: int = Field(
        gt=0,
        description="Maximum accepted size of an individual file in bytes.",
    )

    @field_validator("include", "exclude")
    @classmethod
    def validate_patterns(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Validate repository-relative selection patterns.

        Args:
            values:
                Patterns supplied by configuration.

        Returns:
            The validated patterns.
        """

        return tuple(_validate_repository_pattern(value) for value in values)


class SummaryUnitConfig(_ImmutableConfigModel):
    """Configuration for one explicit logical summary unit.

    Attributes:
        id:
            Stable lowercase kebab-case identifier for the summary unit.
        paths:
            Repository-relative patterns assigning selected files to the unit.
    """

    id: str = Field(
        min_length=1,
        description="Stable lowercase kebab-case summary-unit identifier.",
    )
    paths: tuple[str, ...] = Field(
        min_length=1,
        description="Patterns assigning selected repository files to this unit.",
    )

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        """Validate the logical unit identifier.

        Args:
            value:
                Unit identifier supplied by configuration.

        Returns:
            The validated identifier.
        """

        return _validate_identifier(value, "summary unit id")

    @field_validator("paths")
    @classmethod
    def validate_paths(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Validate the unit's repository-relative patterns.

        Args:
            values:
                Unit patterns supplied by configuration.

        Returns:
            The validated patterns.
        """

        return tuple(_validate_repository_pattern(value) for value in values)


class ProjectConfig(_ImmutableConfigModel):
    """Identity and snapshot settings for one portfolio project.

    Attributes:
        slug:
            Stable lowercase kebab-case project identifier.
        display_name:
            Human-readable project name used in reports and generated content.
        repository:
            Repository provider and immutable-ref settings.
        selection:
            Rules controlling which committed files enter the snapshot.
        summary_units:
            Explicit logical groups to plan from the selected snapshot files.
    """

    slug: str = Field(
        min_length=1,
        description="Stable lowercase kebab-case project identifier.",
    )
    display_name: str = Field(
        min_length=1,
        description="Human-readable project name.",
    )
    repository: RepositoryConfig = Field(
        description="Repository provider and immutable-ref settings."
    )
    selection: SelectionConfig = Field(
        description="Committed-file selection and size-limit settings."
    )
    summary_units: tuple[SummaryUnitConfig, ...] = Field(
        min_length=1,
        description="Explicit logical groups planned from selected files.",
    )

    @field_validator("slug")
    @classmethod
    def validate_slug(cls, value: str) -> str:
        """Validate the normalized project slug.

        Args:
            value:
                Project slug supplied by configuration.

        Returns:
            The validated slug.
        """

        return _validate_identifier(value, "project slug")

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, value: str) -> str:
        """Reject empty or silently normalized display names.

        Args:
            value:
                Display name supplied by configuration.

        Returns:
            The validated display name.

        Raises:
            ValueError:
                If the display name is blank or padded.
        """

        if not value or value != value.strip():
            raise ValueError("display_name must be non-empty and normalized")
        return value

    @model_validator(mode="after")
    def validate_unique_unit_ids(self) -> ProjectConfig:
        """Ensure logical unit identifiers are unique within this project.

        Returns:
            The validated project configuration.

        Raises:
            ValueError:
                If a summary unit identifier occurs more than once.
        """

        unit_ids = [unit.id for unit in self.summary_units]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("summary unit ids must be unique within a project")
        return self


class PortfolioConfig(_ImmutableConfigModel):
    """Top-level versioned portfolio snapshot configuration.

    Attributes:
        version:
            Supported portfolio configuration schema version.
        projects:
            Projects whose committed repository snapshots can be processed.
    """

    version: Literal[1] = Field(description="Portfolio configuration schema version.")
    projects: tuple[ProjectConfig, ...] = Field(
        min_length=1,
        description="Projects configured for committed-snapshot processing.",
    )

    @model_validator(mode="after")
    def validate_unique_project_slugs(self) -> PortfolioConfig:
        """Ensure project slugs are unique across the configuration.

        Returns:
            The validated portfolio configuration.

        Raises:
            ValueError:
                If a project slug occurs more than once.
        """

        slugs = [project.slug for project in self.projects]
        if len(slugs) != len(set(slugs)):
            raise ValueError("project slugs must be unique")
        return self


def _resolve_repository_paths(data: dict[str, Any]) -> None:
    """Resolve local repository paths relative to the application repository.

    Args:
        data:
            Mutable YAML mapping to prepare for Pydantic validation.
    """

    projects = data.get("projects")
    if not isinstance(projects, list):
        return

    for project in projects:
        if not isinstance(project, dict):
            continue
        repository = project.get("repository")
        if not isinstance(repository, dict) or repository.get("type") != "local_git":
            continue
        path_value = repository.get("path")
        if not isinstance(path_value, (str, Path)):
            continue
        repository_path = Path(path_value).expanduser()
        if not repository_path.is_absolute():
            repository_path = settings.REPO_ROOT / repository_path
        repository["path"] = repository_path.resolve()


def load_portfolio_config(path: Path) -> PortfolioConfig:
    """Load and validate a versioned portfolio YAML configuration.

    The configuration file path and every relative local repository path are
    interpreted from :data:`settings.REPO_ROOT`, not from the process working
    directory or the configuration file's directory.

    Args:
        path:
            YAML configuration file to load.

    Returns:
        An immutable validated portfolio configuration.

    Raises:
        FileNotFoundError:
            If the configuration file does not exist.
        ValueError:
            If YAML is malformed or the document root is not a mapping.
        pydantic.ValidationError:
            If the schema version, keys, or values are invalid.
    """

    config_path = path.expanduser()
    if not config_path.is_absolute():
        config_path = settings.REPO_ROOT / config_path

    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Malformed YAML configuration: {config_path}") from error

    if not isinstance(loaded, dict):
        raise ValueError(  # noqa: TRY004 - public loader contract uses ValueError.
            "Portfolio configuration must be a YAML mapping"
        )

    _resolve_repository_paths(loaded)
    return PortfolioConfig.model_validate(loaded)
