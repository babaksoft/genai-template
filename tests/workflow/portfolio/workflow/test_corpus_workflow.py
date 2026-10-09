"""End-to-end tests for the Stage 1 Portfolio corpus workflow."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from decimal import Decimal
from pathlib import Path

import pytest

from genai_template.workflow.portfolio.config.models import (
    GenerationConfig,
    GenerationInputLimits,
    GenerationLocations,
    InferenceConfig,
    LocalGitRepositoryConfig,
    PortfolioConfig,
    ProjectConfig,
    ProjectDocumentContextConfig,
    SelectionConfig,
    StructuredGenerationConfig,
    SummaryUnitConfig,
    TokenPricingConfig,
)
from genai_template.workflow.portfolio.domain import (
    GenerationRequest,
    ProviderAuditMetadata,
    TextGenerationError,
    TextGenerationResponse,
    TokenUsage,
)
from genai_template.workflow.portfolio.infrastructure.generation import (
    SUMMARY_SPECIFICATIONS,
)
from genai_template.workflow.portfolio.infrastructure.repositories import (
    LocalGitSnapshotReader,
)
from genai_template.workflow.portfolio.workflow import (
    PortfolioCorpusWorkflow,
    run_portfolio_corpus_workflow,
)


class _CountingGenerator:
    """Deterministic generator returning evidence-scoped typed summaries.

    Attributes:
        calls:
            Generation fingerprints received in provider-call order.
    """

    def __init__(self) -> None:
        """Initialize an empty provider-call record."""

        self.calls: list[str] = []

    def generate(self, request: GenerationRequest) -> TextGenerationResponse:
        """Return valid Markdown for the requested artifact contract.

        Args:
            request:
                Fully identified generation request.
        Returns:
            Deterministic plain-text response with complete accounting.
        """

        self.calls.append(request.provenance.generation_fingerprint)
        specification = SUMMARY_SPECIFICATIONS[request.provenance.artifact_kind]
        text = "\n\n".join(
            f"## {section.heading}\n\nDeterministic test summary.\n\n"
            f"### Evidence\n\n- {request.input_paths[0]}"
            for section in specification.sections
        )
        return TextGenerationResponse(
            text=text,
            provider="ollama",
            model="test-model",
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
            provider_metadata=ProviderAuditMetadata(request_id="test-request"),
        )


class _FailingProjectGenerator:
    """Provider boundary that fails after regenerating the changed component.

    Attributes:
        delegate:
            Deterministic generator used for component requests.
    """

    def __init__(self) -> None:
        """Initialize the successful component delegate."""

        self.delegate = _CountingGenerator()

    def generate(self, request: GenerationRequest) -> TextGenerationResponse:
        """Generate components and fail the first project synthesis request.

        Args:
            request:
                Fully identified generation request.

        Raises:
            TextGenerationError:
                When project synthesis reaches the provider boundary.
        """

        if request.provenance.artifact_kind == "component":
            return self.delegate.generate(request)
        raise TextGenerationError(
            "provider request failed",
            provider="ollama",
            model="test-model",
            reason="transport-error",
        )


def _git(repository: Path, *arguments: str) -> str:
    """Run Git in a temporary test repository.

    Args:
        repository:
            Test repository path.
        *arguments:
            Git arguments.

    Returns:
        Stripped standard output.
    """

    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        env=os.environ.copy(),
        text=True,
    )
    return result.stdout.strip()


def _create_repository(path: Path) -> None:
    """Create a small committed repository covering every synthesis context.

    Args:
        path:
            Directory to initialize and populate.
    """

    path.mkdir()
    _git(path, "init", "--initial-branch=main")
    (path / "README.md").write_text("# Sample\n", encoding="utf-8")
    source = path / "src"
    source.mkdir()
    (source / "app.py").write_text('"""Sample app."""\n', encoding="utf-8")
    tests = path / "tests"
    tests.mkdir()
    (tests / "test_app.py").write_text('"""Sample tests."""\n', encoding="utf-8")
    _git(path, "add", "--all")
    _git(
        path,
        "-c",
        "user.name=Portfolio Tests",
        "-c",
        "user.email=portfolio@example.test",
        "commit",
        "--message=initial",
    )


def _config(repository: Path, output_root: Path) -> PortfolioConfig:
    """Build a valid injected config using temporary output locations.

    Args:
        repository:
            Local Git test repository.
        output_root:
            Root for isolated cache and publication data.

    Returns:
        Complete balanced Portfolio configuration.
    """

    locations = GenerationLocations.model_construct(
        cache=output_root / "cache",
        publication=output_root / "data" / "portfolio",
    )
    generation = GenerationConfig.model_construct(
        profile="balanced",
        structured_generation=StructuredGenerationConfig(
            provider="ollama",
            model="test-model",
            inference=InferenceConfig(temperature=0, seed=1),
        ),
        prompt_version="v1",
        output_schema_version="v1",
        project_context=ProjectDocumentContextConfig(
            overview=("README.md",),
            architecture=("src/**/*.py",),
            testing_operations=("tests/**/*.py",),
        ),
        input_limits=GenerationInputLimits(max_files=10, max_bytes=100_000),
        locations=locations,
        pricing=TokenPricingConfig(
            input_per_million_tokens=Decimal(1),
            output_per_million_tokens=Decimal(2),
        ),
    )
    return PortfolioConfig.model_construct(
        version=1,
        generation=generation,
        projects=(
            ProjectConfig(
                slug="sample",
                display_name="Sample Project",
                repository=LocalGitRepositoryConfig(
                    type="local_git",
                    path=repository,
                    ref="HEAD",
                ),
                selection=SelectionConfig(
                    include=("README.md", "src/**/*.py", "tests/**/*.py"),
                    max_file_bytes=10_000,
                ),
                summary_units=(
                    SummaryUnitConfig(
                        id="application",
                        paths=("src/**/*.py", "tests/**/*.py"),
                    ),
                ),
            ),
        ),
    )


def test_unchanged_second_run_uses_only_validated_cache(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A real snapshot/cache/publisher run is byte-identical without new calls."""

    caplog.set_level(logging.INFO)
    repository = tmp_path / "repository"
    _create_repository(repository)
    config = _config(repository, tmp_path / "outputs")
    generator = _CountingGenerator()
    workflow = PortfolioCorpusWorkflow(
        repository_reader=LocalGitSnapshotReader(),
        generator_factory=lambda generation: generator,
        config_loader=lambda path: config,
    )

    first = asyncio.run(
        run_portfolio_corpus_workflow(
            workflow,
            config_path=Path("unused.yml"),
            project_slug="sample",
        )
    )
    first_call_count = len(generator.calls)
    assert first.release_path is not None
    first_manifest = (first.release_path / "manifest.json").read_bytes()
    second = asyncio.run(
        run_portfolio_corpus_workflow(
            workflow,
            config_path=Path("unused.yml"),
            project_slug="sample",
        )
    )

    assert first_call_count == 4
    assert len(generator.calls) == first_call_count
    assert first.provider_call_count == 4
    assert second.provider_call_count == 0
    assert all(artifact.cache_hit for artifact in second.artifacts)
    assert all(
        artifact.original_token_usage == TokenUsage(input_tokens=10, output_tokens=5)
        for artifact in second.artifacts
    )
    assert second.billed_token_usage == TokenUsage(
        input_tokens=0,
        output_tokens=0,
    )
    assert second.corpus_fingerprint == first.corpus_fingerprint
    assert second.release_path == first.release_path
    assert second.release_path is not None
    assert second.published_path is not None
    assert (second.release_path / "manifest.json").read_bytes() == first_manifest
    assert second.published_path.is_symlink()
    assert {path.name for path in second.release_path.iterdir()} == {
        "manifest.json",
        "sample--overview.md",
        "sample--architecture.md",
        "sample--testing-operations.md",
        "sample--component--application.md",
    }
    assert tuple(step.name for step in second.steps) == (
        "configuration",
        "snapshot",
        "planning",
        "components",
        "project-synthesis",
        "rendering",
        "manifest-validation",
        "publication",
    )
    assert "Portfolio workflow step started: configuration" in caplog.text
    assert "Portfolio workflow step completed: publication" in caplog.text
    assert "Processing component summary application" in caplog.text
    assert "Project summary overview generated and cached" in caplog.text


def test_unknown_project_fails_before_repository_or_publication(
    tmp_path: Path,
) -> None:
    """Project selection failure creates no cache, release, or public pointer."""

    repository = tmp_path / "repository"
    _create_repository(repository)
    output_root = tmp_path / "outputs"
    config = _config(repository, output_root)
    workflow = PortfolioCorpusWorkflow(
        repository_reader=LocalGitSnapshotReader(),
        generator_factory=lambda generation: _CountingGenerator(),
        config_loader=lambda path: config,
    )

    with pytest.raises(ValueError, match="configured project not found"):
        asyncio.run(
            run_portfolio_corpus_workflow(
                workflow,
                config_path=Path("unused.yml"),
                project_slug="missing",
            )
        )

    assert not output_root.exists()


def test_provider_failure_leaves_previous_publication_untouched(
    tmp_path: Path,
) -> None:
    """A failed replacement run cannot switch or mutate the public release."""

    repository = tmp_path / "repository"
    _create_repository(repository)
    output_root = tmp_path / "outputs"
    config = _config(repository, output_root)
    successful_workflow = PortfolioCorpusWorkflow(
        repository_reader=LocalGitSnapshotReader(),
        generator_factory=lambda generation: _CountingGenerator(),
        config_loader=lambda path: config,
    )
    first = asyncio.run(
        run_portfolio_corpus_workflow(
            successful_workflow,
            config_path=Path("unused.yml"),
            project_slug="sample",
        )
    )
    assert first.published_path is not None
    assert first.release_path is not None
    pointer_before = os.readlink(first.published_path)
    manifest_before = (first.release_path / "manifest.json").read_bytes()

    (repository / "src" / "app.py").write_text(
        '"""Changed sample app."""\n', encoding="utf-8"
    )
    _git(repository, "add", "src/app.py")
    _git(
        repository,
        "-c",
        "user.name=Portfolio Tests",
        "-c",
        "user.email=portfolio@example.test",
        "commit",
        "--message=change-source",
    )
    failing_workflow = PortfolioCorpusWorkflow(
        repository_reader=LocalGitSnapshotReader(),
        generator_factory=lambda generation: _FailingProjectGenerator(),
        config_loader=lambda path: config,
    )

    with pytest.raises(TextGenerationError, match="provider request failed"):
        asyncio.run(
            run_portfolio_corpus_workflow(
                failing_workflow,
                config_path=Path("unused.yml"),
                project_slug="sample",
            )
        )

    assert os.readlink(first.published_path) == pointer_before
    assert (first.release_path / "manifest.json").read_bytes() == manifest_before
