"""YAML loading for validated Portfolio snapshot configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from genai_template.config import settings
from genai_template.workflow.portfolio.config.models import PortfolioConfig


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


def _resolve_generation_paths(data: dict[str, Any]) -> None:
    """Resolve configured output paths without permitting traversal.

    Args:
        data:
            Mutable YAML mapping to prepare for Pydantic validation.

    Raises:
        ValueError:
            If an output path is absolute, traversing, or malformed.
    """

    generation = data.get("generation")
    if not isinstance(generation, dict):
        return
    locations = generation.get("locations")
    if not isinstance(locations, dict):
        return
    for name in ("cache", "publication"):
        raw_path = locations.get(name)
        if not isinstance(raw_path, (str, Path)):
            continue
        path = Path(raw_path)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError(
                f"generation {name} path must be repository-root-relative and safe"
            )
        locations[name] = (settings.REPO_ROOT / path).absolute()


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
    _resolve_generation_paths(loaded)
    return PortfolioConfig.model_validate(loaded)
