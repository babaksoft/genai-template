"""Tests for Portfolio balanced-generation configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from genai_template.config import settings
from genai_template.workflow.portfolio.config import (
    GenerationConfig,
    GenerationInputLimits,
    GenerationLocations,
    InferenceConfig,
    ProjectDocumentContextConfig,
    StructuredGenerationConfig,
    TokenPricingConfig,
    load_portfolio_config,
)


def _balanced_config() -> dict[str, Any]:
    """Return a complete balanced one-project configuration.

    Returns:
        Independent mutable configuration data.
    """

    return {
        "version": 1,
        "generation": {
            "profile": "balanced",
            "structured_generation": {
                "provider": "ollama",
                "model": "llama3.2",
                "inference": {"temperature": 0.2, "seed": 11},
                "timeout_seconds": 30,
            },
            "prompt_version": "v1",
            "output_schema_version": "v1",
            "project_context": {
                "overview": ["README.md"],
                "architecture": ["src/**/*.py"],
                "testing_operations": ["tests/**/*.py"],
            },
            "input_limits": {"max_files": 50, "max_bytes": 100_000},
            "locations": {
                "cache": "storage/portfolio-cache",
                "publication": "data/portfolio",
            },
            "pricing": {
                "input_per_million_tokens": "0.10",
                "output_per_million_tokens": "0.30",
            },
        },
        "projects": [
            {
                "slug": "sample-project",
                "display_name": "Sample Project",
                "repository": {"type": "local_git", "path": ".", "ref": "HEAD"},
                "selection": {
                    "include": ["README.md", "src/**/*.py"],
                    "exclude": [],
                    "max_file_bytes": 100_000,
                },
                "summary_units": [{"id": "core", "paths": ["src/**/*.py"]}],
            }
        ],
    }


def _write(tmp_path: Path, data: object) -> Path:
    """Write a temporary YAML profile.

    Args:
        tmp_path:
            Temporary directory supplied by pytest.
        data:
            YAML-compatible data to write.

    Returns:
        Path of the written profile.
    """

    path = tmp_path / "portfolio.yml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_snapshot_only_configuration_remains_valid(tmp_path: Path) -> None:
    """Stage 0 profiles can continue omitting generation settings."""

    data = _balanced_config()
    data.pop("generation")

    config = load_portfolio_config(_write(tmp_path, data))

    assert config.generation is None
    with pytest.raises(ValueError, match="generation settings are required"):
        config.require_generation()


def test_balanced_configuration_is_typed_resolved_and_immutable(
    tmp_path: Path,
) -> None:
    """Valid generation settings are typed and rooted at the repository."""

    generation = load_portfolio_config(
        _write(tmp_path, _balanced_config())
    ).require_generation()

    assert generation.structured_generation.provider == "ollama"
    assert generation.locations.cache == settings.REPO_ROOT / "storage/portfolio-cache"
    assert generation.locations.publication == settings.REPO_ROOT / "data/portfolio"
    with pytest.raises(ValidationError):
        cast(Any, generation).profile = "changed"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("generation", "unexpected"), True),
        (("generation", "profile"), "deep"),
        (("generation", "structured_generation", "provider"), "anthropic"),
        (("generation", "structured_generation", "inference", "temperature"), -0.1),
        (("generation", "structured_generation", "inference", "seed"), -1),
        (("generation", "pricing", "input_per_million_tokens"), -0.01),
        (("generation", "pricing", "output_per_million_tokens"), "not-a-rate"),
        (("generation", "locations", "cache"), "../cache"),
        (("generation", "locations", "publication"), "/tmp/portfolio"),
    ],
    ids=[
        "unknown-field",
        "unsupported-profile",
        "unsupported-provider",
        "invalid-temperature",
        "invalid-seed",
        "negative-rate",
        "malformed-rate",
        "traversing-cache",
        "absolute-publication",
    ],
)
def test_invalid_generation_configuration_is_rejected(
    tmp_path: Path, path: tuple[str, ...], value: object
) -> None:
    """Generation schema, provider, pricing, and path violations fail early."""

    data = deepcopy(_balanced_config())
    target: dict[str, Any] = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises((ValidationError, ValueError)):
        load_portfolio_config(_write(tmp_path, data))


def test_provider_incompatible_seed_is_rejected(tmp_path: Path) -> None:
    """The OpenAI adapter rejects deterministic seed settings it cannot honor."""

    data = _balanced_config()
    data["generation"]["structured_generation"]["provider"] = "openai"

    with pytest.raises(ValidationError, match="seed is not supported"):
        load_portfolio_config(_write(tmp_path, data))


def test_credentials_are_not_configuration_fields(tmp_path: Path) -> None:
    """Provider credentials cannot enter validated generation configuration."""

    data = _balanced_config()
    data["generation"]["structured_generation"]["api_key"] = "secret"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_portfolio_config(_write(tmp_path, data))


@pytest.mark.parametrize(
    "model_type",
    [
        InferenceConfig,
        StructuredGenerationConfig,
        ProjectDocumentContextConfig,
        GenerationInputLimits,
        GenerationLocations,
        TokenPricingConfig,
        GenerationConfig,
    ],
)
def test_generation_models_document_classes_and_fields(
    model_type: type[BaseModel],
) -> None:
    """Every generation configuration model documents its public contract."""

    assert model_type.__doc__ is not None
    assert "Attributes:" in model_type.__doc__
    assert all(field.description for field in model_type.model_fields.values())
