"""Opt-in real-provider tests for the Portfolio Markdown prompt contracts."""

from __future__ import annotations

import asyncio
import hashlib
import os
import subprocess
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Literal

import pytest

from genai_template.workflow.portfolio.cli.generation import render_json_report
from genai_template.workflow.portfolio.config import (
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
    ArtifactProvenance,
    ComponentSummary,
    GenerationRequest,
    GenerationRunReport,
    ProjectOverviewSummary,
    SnapshotFile,
)
from genai_template.workflow.portfolio.domain.contracts import TextGenerator
from genai_template.workflow.portfolio.generation import (
    COMPONENT_PROMPT,
    OVERVIEW_PROMPT,
    assemble_component_prompt,
    assemble_project_prompt,
    generate_validated_summary,
)
from genai_template.workflow.portfolio.infrastructure.generators import (
    OllamaTextGenerator,
    OpenAITextGenerator,
)
from genai_template.workflow.portfolio.infrastructure.repositories import (
    LocalGitSnapshotReader,
)
from genai_template.workflow.portfolio.workflow import (
    PortfolioCorpusWorkflow,
    run_portfolio_corpus_workflow,
)

_COMPLETION_GATE_ENABLED = os.getenv("PORTFOLIO_RUN_COMPLETION_GATE") == "true"


def _snapshot_file(path: str, text: str) -> SnapshotFile:
    """Build one canonical source file for a representative live prompt.

    Args:
        path:
            Repository-relative evidence path.
        text:
            Small source document supplied to the provider.

    Returns:
        Immutable snapshot file with matching byte metadata.
    """

    encoded = text.encode("utf-8")
    return SnapshotFile(
        path=path,
        text=text,
        content_hash=hashlib.sha256(encoded).hexdigest(),
        byte_size=len(encoded),
    )


def _provenance(
    provider: Literal["ollama", "openai"],
    model: str,
    *,
    artifact_kind: Literal["component", "overview"],
) -> ArtifactProvenance:
    """Build stable identity for one representative live-provider request.

    Args:
        provider:
            Provider selected by the integration test.
        model:
            Explicit model selected through the environment.
        artifact_kind:
            Prompt contract exercised by the request.

    Returns:
        Valid provenance matching the selected adapter.
    """

    definition = COMPONENT_PROMPT if artifact_kind == "component" else OVERVIEW_PROMPT
    return ArtifactProvenance(
        artifact_kind=artifact_kind,
        generation_fingerprint="1" * 64,
        source_fingerprint="2" * 64,
        unit_input_fingerprint="3" * 64 if artifact_kind == "component" else None,
        component_artifact_hashes=("4" * 64,) if artifact_kind == "overview" else (),
        prompt_id=definition.prompt_id,
        prompt_hash=definition.prompt_hash,
        output_schema_version="v1",
        provider=provider,
        model=model,
        temperature=0,
        seed=7 if provider == "ollama" else None,
    )


def _assert_real_prompt_contracts(
    generator: TextGenerator,
    provider: Literal["ollama", "openai"],
    model: str,
) -> None:
    """Exercise one component and one project contract through a real provider.

    Args:
        generator:
            Configured official-SDK provider adapter.
        provider:
            Expected provider identity.
        model:
            Expected model identity.
    """

    component_file = _snapshot_file(
        "src/example.py",
        (
            '"""Small arithmetic component."""\n\n'
            "def add(left: int, right: int) -> int:\n"
            '    """Return the sum of two integers."""\n\n'
            "    return left + right\n"
        ),
    )
    component_request = GenerationRequest(
        provenance=_provenance(
            provider,
            model,
            artifact_kind="component",
        ),
        prompt=assemble_component_prompt(COMPONENT_PROMPT, (component_file,)),
        input_paths=(component_file.path,),
    )
    component = generate_validated_summary(
        generator,
        component_request,
        ComponentSummary,
        repository_context_paths=(component_file.path,),
    )

    assert component.warnings == ()

    readme_file = _snapshot_file(
        "README.md",
        (
            "# Example Project\n\n"
            "A typed Python arithmetic library exposing `add` from "
            "`src/example.py`.\n"
        ),
    )
    overview_request = GenerationRequest(
        provenance=_provenance(provider, model, artifact_kind="overview"),
        prompt=assemble_project_prompt(
            OVERVIEW_PROMPT,
            (readme_file,),
            {"arithmetic": component.value},
        ),
        input_paths=(readme_file.path,),
    )
    overview = generate_validated_summary(
        generator,
        overview_request,
        ProjectOverviewSummary,
        repository_context_paths=(readme_file.path,),
        component_evidence_paths=(component_file.path,),
    )

    assert overview.warnings == ()


def _git(repository: Path, *arguments: str) -> str:
    """Run Git for the isolated completion-gate repository.

    Args:
        repository:
            Temporary repository path.
        *arguments:
            Git command arguments.

    Returns:
        Stripped command output.
    """

    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        env=os.environ.copy(),
        text=True,
    )
    return result.stdout.strip()


def _create_gate_repository(repository: Path) -> None:
    """Create one committed project covering every corpus artifact kind.

    Args:
        repository:
            Fresh temporary repository path.
    """

    repository.mkdir()
    _git(repository, "init", "--initial-branch=main")
    (repository / "README.md").write_text(
        "# Calculator\n\nA typed Python library for integer addition.\n",
        encoding="utf-8",
    )
    source = repository / "src"
    source.mkdir()
    (source / "calculator.py").write_text(
        (
            '"""Integer arithmetic."""\n\n'
            "def add(left: int, right: int) -> int:\n"
            '    """Return the sum of two integers."""\n\n'
            "    return left + right\n"
        ),
        encoding="utf-8",
    )
    tests = repository / "tests"
    tests.mkdir()
    (tests / "test_calculator.py").write_text(
        (
            '"""Tests for integer arithmetic."""\n\n'
            "from calculator import add\n\n\n"
            "def test_adds_integers() -> None:\n"
            "    assert add(2, 3) == 5\n"
        ),
        encoding="utf-8",
    )
    _git(repository, "add", "--all")
    _git(
        repository,
        "-c",
        "user.name=Portfolio Completion Gate",
        "-c",
        "user.email=portfolio-gate@example.test",
        "commit",
        "--message=completion-gate-snapshot",
    )


def _gate_config(
    repository: Path,
    output_root: Path,
    provider: Literal["ollama", "openai"],
    model: str,
) -> PortfolioConfig:
    """Build an isolated real-provider one-project workflow configuration.

    Args:
        repository:
            Committed representative project repository.
        output_root:
            Temporary cache and publication parent.
        provider:
            Provider path under test.
        model:
            Explicit provider model.

    Returns:
        Complete injected workflow configuration.
    """

    generation = GenerationConfig.model_construct(
        profile="balanced",
        structured_generation=StructuredGenerationConfig(
            provider=provider,
            model=model,
            inference=InferenceConfig(
                temperature=0,
                seed=7 if provider == "ollama" else None,
            ),
            timeout_seconds=180,
        ),
        prompt_version="v1",
        output_schema_version="v1",
        project_context=ProjectDocumentContextConfig(
            overview=("README.md",),
            architecture=("src/**/*.py",),
            testing_operations=("README.md", "tests/**/*.py"),
        ),
        input_limits=GenerationInputLimits(max_files=10, max_bytes=100_000),
        locations=GenerationLocations.model_construct(
            cache=output_root / "cache",
            publication=output_root / "data" / "portfolio",
        ),
        pricing=TokenPricingConfig(
            input_per_million_tokens=Decimal(0),
            output_per_million_tokens=Decimal(0),
        ),
    )
    return PortfolioConfig.model_construct(
        version=1,
        generation=generation,
        projects=(
            ProjectConfig(
                slug="calculator",
                display_name="Calculator",
                repository=LocalGitRepositoryConfig(
                    type="local_git",
                    path=repository,
                    ref="HEAD",
                ),
                selection=SelectionConfig(
                    include=("README.md", "src/**/*.py", "tests/**/*.py"),
                    max_file_bytes=100_000,
                ),
                summary_units=(
                    SummaryUnitConfig(
                        id="calculator",
                        paths=("src/**/*.py", "tests/**/*.py"),
                    ),
                ),
            ),
        ),
    )


def _preserve_reports(
    provider: Literal["ollama", "openai"],
    first: GenerationRunReport,
    second: GenerationRunReport,
) -> None:
    """Optionally preserve content-safe JSON reports outside pytest temp data.

    Args:
        provider:
            Provider represented by the reports.
        first:
            Empty-cache workflow report.
        second:
            Unchanged cache-hit workflow report.
    """

    configured_path = os.getenv("PORTFOLIO_COMPLETION_EVIDENCE_DIR")
    if configured_path is None:
        return
    evidence_path = Path(configured_path).expanduser()
    evidence_path.mkdir(parents=True, exist_ok=True)
    (evidence_path / f"{provider}-first-run.json").write_text(
        f"{render_json_report(first)}\n",
        encoding="utf-8",
    )
    (evidence_path / f"{provider}-cache-hit.json").write_text(
        f"{render_json_report(second)}\n",
        encoding="utf-8",
    )


def _run_completion_gate(
    tmp_path: Path,
    provider: Literal["ollama", "openai"],
    model: str,
    generator_factory: Callable[[GenerationConfig], TextGenerator],
) -> None:
    """Publish from an empty cache and prove unchanged cache reuse.

    Args:
        tmp_path:
            Fresh pytest directory isolating source, cache, and publication.
        provider:
            Provider path under test.
        model:
            Explicit provider model.
        generator_factory:
            Callable returning the configured provider adapter.
    """

    repository = tmp_path / "repository"
    _create_gate_repository(repository)
    config = _gate_config(repository, tmp_path / "output", provider, model)
    workflow = PortfolioCorpusWorkflow(
        repository_reader=LocalGitSnapshotReader(),
        generator_factory=generator_factory,
        config_loader=lambda _: config,
    )

    first = asyncio.run(
        run_portfolio_corpus_workflow(
            workflow,
            config_path=tmp_path / "injected.yml",
            project_slug="calculator",
        )
    )
    second = asyncio.run(
        run_portfolio_corpus_workflow(
            workflow,
            config_path=tmp_path / "injected.yml",
            project_slug="calculator",
        )
    )

    assert first.provider_call_count == 4
    assert all(not artifact.cache_hit for artifact in first.artifacts)
    assert all(not artifact.generation_warning_counts for artifact in first.artifacts)
    assert first.published_path is not None and first.published_path.is_symlink()
    assert first.release_path is not None and first.release_path.is_dir()
    assert second.provider_call_count == 0
    assert all(artifact.cache_hit for artifact in second.artifacts)
    assert second.corpus_fingerprint == first.corpus_fingerprint
    _preserve_reports(provider, first, second)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or not os.getenv("PORTFOLIO_OPENAI_MODEL"),
    reason="OPENAI_API_KEY and PORTFOLIO_OPENAI_MODEL are required.",
)
def test_openai_honors_component_and_project_prompt_contracts() -> None:
    """OpenAI should return warning-free summaries for both real prompt shapes."""

    model = os.environ["PORTFOLIO_OPENAI_MODEL"]
    generator = OpenAITextGenerator(model, temperature=0, timeout_seconds=180)

    _assert_real_prompt_contracts(generator, "openai", model)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("PORTFOLIO_OLLAMA_MODEL"),
    reason="PORTFOLIO_OLLAMA_MODEL is required.",
)
def test_ollama_cloud_honors_component_and_project_prompt_contracts() -> None:
    """Ollama Cloud should honor both real prompt shapes without recovery."""

    model = os.environ["PORTFOLIO_OLLAMA_MODEL"]
    generator = OllamaTextGenerator(
        model,
        temperature=0,
        seed=7,
        timeout_seconds=180,
    )

    _assert_real_prompt_contracts(generator, "ollama", model)


@pytest.mark.integration
@pytest.mark.skipif(
    not _COMPLETION_GATE_ENABLED
    or not os.getenv("OPENAI_API_KEY")
    or not os.getenv("PORTFOLIO_OPENAI_MODEL"),
    reason=(
        "PORTFOLIO_RUN_COMPLETION_GATE=true, OPENAI_API_KEY, and "
        "PORTFOLIO_OPENAI_MODEL are required."
    ),
)
def test_openai_completes_and_reuses_one_project_workflow(tmp_path: Path) -> None:
    """OpenAI should publish once and reuse all four unchanged artifacts."""

    model = os.environ["PORTFOLIO_OPENAI_MODEL"]

    def generator_factory(_: GenerationConfig) -> TextGenerator:
        """Create the OpenAI adapter selected by this completion gate.

        Args:
            _:
                Injected generation configuration already represented by closure.

        Returns:
            Configured OpenAI text generator.
        """

        return OpenAITextGenerator(model, temperature=0, timeout_seconds=180)

    _run_completion_gate(tmp_path, "openai", model, generator_factory)


@pytest.mark.integration
@pytest.mark.skipif(
    not _COMPLETION_GATE_ENABLED or not os.getenv("PORTFOLIO_OLLAMA_MODEL"),
    reason=(
        "PORTFOLIO_RUN_COMPLETION_GATE=true and PORTFOLIO_OLLAMA_MODEL " "are required."
    ),
)
def test_ollama_cloud_completes_and_reuses_one_project_workflow(
    tmp_path: Path,
) -> None:
    """Ollama Cloud should publish once and reuse all four unchanged artifacts."""

    model = os.environ["PORTFOLIO_OLLAMA_MODEL"]

    def generator_factory(_: GenerationConfig) -> TextGenerator:
        """Create the Ollama Cloud adapter selected by this completion gate.

        Args:
            _:
                Injected generation configuration already represented by closure.

        Returns:
            Configured Ollama Cloud text generator.
        """

        return OllamaTextGenerator(
            model,
            temperature=0,
            seed=7,
            timeout_seconds=180,
        )

    _run_completion_gate(tmp_path, "ollama", model, generator_factory)
