"""Tests for cached project-document synthesis."""

from __future__ import annotations

import hashlib
from decimal import Decimal
from pathlib import Path

import pytest

from genai_template.config import settings
from genai_template.workflow.portfolio.domain import (
    ArtifactProvenance,
    ArtifactRunReport,
    ArtifactValidationError,
    CachedArtifact,
    ComponentSummary,
    ComponentSummaryArtifact,
    EvidenceSection,
    GenerationRequest,
    ProviderAuditMetadata,
    RepositorySnapshot,
    SnapshotFile,
    TextGenerationResponse,
    TokenUsage,
)
from genai_template.workflow.portfolio.domain.config import (
    GenerationConfig,
    GenerationInputLimits,
    GenerationLocations,
    InferenceConfig,
    ProjectDocumentContextConfig,
    StructuredGenerationConfig,
    TokenPricingConfig,
)
from genai_template.workflow.portfolio.infrastructure.caches import (
    CACHE_SCHEMA_VERSION,
    FilesystemArtifactCache,
)
from genai_template.workflow.portfolio.infrastructure.fingerprints import (
    sha256_canonical_json,
)
from genai_template.workflow.portfolio.infrastructure.generation import (
    SUMMARY_SPECIFICATIONS,
    generate_project_summaries,
)


class _ProjectGenerator:
    """Deterministic project generator recording requested artifact kinds.

    Attributes:
        calls:
            Project artifact kinds requested in call order.
    """

    def __init__(self) -> None:
        """Initialize an empty call record."""

        self.calls: list[str] = []

    def generate(self, request: GenerationRequest) -> TextGenerationResponse:
        """Return every requested section with valid repository evidence.

        Args:
            request:
                Project synthesis request.
        Returns:
            Deterministic plain-text provider response.
        """

        self.calls.append(request.provenance.artifact_kind)
        specification = SUMMARY_SPECIFICATIONS[request.provenance.artifact_kind]
        text = "\n\n".join(
            f"## {section.heading}\n\n"
            f"Generated {request.provenance.artifact_kind}.\n\n"
            f"### Evidence\n\n- {request.input_paths[0]}"
            for section in specification.sections
        )
        return TextGenerationResponse(
            text=text,
            provider="ollama",
            model="test-model",
            token_usage=TokenUsage(input_tokens=50, output_tokens=10),
            provider_metadata=ProviderAuditMetadata(request_id="project-request"),
        )


class _RecoveringProjectGenerator:
    """Return malformed but recoverable Markdown for every project artifact."""

    def generate(self, request: GenerationRequest) -> TextGenerationResponse:
        """Return text exercising every deterministic parser warning category.

        Args:
            request:
                Project synthesis request.

        Returns:
            Malformed non-empty Markdown with one valid evidence citation.
        """

        sections = SUMMARY_SPECIFICATIONS[request.provenance.artifact_kind].sections
        first = sections[0].heading
        second = sections[1].heading
        return TextGenerationResponse(
            text=(
                "Provider preamble.\n\n"
                f"## {second}\n\nSecond section.\n\n"
                "### Evidence\n\n- outside/scope.py\n\n"
                f"## {first}\n\nPrimary section.\n\n"
                f"### Evidence\n\n- {request.input_paths[0]}\n\n"
                f"## {first}\n\nDuplicate primary section.\n\n"
                "## Unexpected\n\nUnexpected epilogue."
            ),
            provider="ollama",
            model="test-model",
            token_usage=TokenUsage(input_tokens=50, output_tokens=10),
        )


def _file(path: str, text: str) -> SnapshotFile:
    """Build one canonical snapshot file.

    Args:
        path:
            Repository-relative path.
        text:
            Normalized source text.

    Returns:
        Canonical snapshot file.
    """

    encoded = text.encode("utf-8")
    return SnapshotFile(
        path=path,
        text=text,
        content_hash=hashlib.sha256(encoded).hexdigest(),
        byte_size=len(encoded),
    )


def _snapshot() -> RepositorySnapshot:
    """Build a snapshot covering all project contexts.

    Returns:
        Canonical test snapshot.
    """

    return RepositorySnapshot(
        project_slug="sample",
        requested_ref="HEAD",
        resolved_commit_sha="a" * 40,
        source_fingerprint="b" * 64,
        files=(
            _file("README.md", "overview\n"),
            _file("src/api.py", "architecture\n"),
            _file("tests/test_api.py", "tests\n"),
        ),
    )


def _config(
    *,
    max_bytes: int = 100_000,
    contexts: ProjectDocumentContextConfig | None = None,
) -> GenerationConfig:
    """Build balanced generation settings.

    Args:
        max_bytes:
            Complete source and component byte budget per call.
        contexts:
            Optional project-context pattern override.

    Returns:
        Valid generation configuration.
    """

    return GenerationConfig(
        profile="balanced",
        structured_generation=StructuredGenerationConfig(
            provider="ollama",
            model="test-model",
            inference=InferenceConfig(temperature=0, seed=7),
        ),
        prompt_version="v1",
        output_schema_version="v1",
        project_context=contexts
        or ProjectDocumentContextConfig(
            overview=("README.md",),
            architecture=("src/**/*.py",),
            testing_operations=("tests/**/*.py",),
        ),
        input_limits=GenerationInputLimits(max_files=10, max_bytes=max_bytes),
        locations=GenerationLocations(
            cache=settings.REPO_ROOT / "storage/test-project-cache",
            publication=settings.REPO_ROOT / "data/test-project-corpus",
        ),
        pricing=TokenPricingConfig(
            input_per_million_tokens=Decimal(2),
            output_per_million_tokens=Decimal(8),
        ),
    )


def _component(content: str = "Component behavior.") -> ComponentSummaryArtifact:
    """Build one internally consistent validated component artifact.

    Args:
        content:
            Component statement used to test dependency invalidation.

    Returns:
        Valid component artifact.
    """

    section = EvidenceSection(
        content=(content,),
        evidence_paths=("src/api.py",),
    )
    summary = ComponentSummary(
        responsibilities=section,
        important_abstractions=section,
        behavior=section,
        constraints=section,
        testing_evidence=section,
    )
    output = summary.model_dump(mode="json")
    output_hash = sha256_canonical_json(output)
    fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()
    provenance = ArtifactProvenance(
        artifact_kind="component",
        generation_fingerprint=fingerprint,
        source_fingerprint="b" * 64,
        unit_input_fingerprint="c" * 64,
        prompt_id="portfolio-component-v1",
        prompt_hash="d" * 64,
        output_schema_version="v1",
        provider="ollama",
        model="test-model",
        temperature=0,
        seed=7,
    )
    artifact = CachedArtifact(
        cache_schema_version=CACHE_SCHEMA_VERSION,
        provenance=provenance,
        summary=output,
        output_hash=output_hash,
        raw_response_hash="e" * 64,
        original_token_usage=TokenUsage(input_tokens=5, output_tokens=2),
    )
    return ComponentSummaryArtifact(
        unit_id="api",
        summary=summary,
        artifact=artifact,
        run_report=ArtifactRunReport(
            artifact_kind="component",
            generation_fingerprint=fingerprint,
            cache_hit=False,
            token_usage=artifact.original_token_usage,
            latency_seconds=0,
        ),
    )


def test_unchanged_project_synthesis_uses_cache_without_provider_calls(
    tmp_path: Path,
) -> None:
    """All three project documents are reused on a fully unchanged rerun."""

    cache = FilesystemArtifactCache(tmp_path / "cache")
    first_generator = _ProjectGenerator()
    first = generate_project_summaries(
        _snapshot(), (_component(),), _config(), first_generator, cache
    )
    second_generator = _ProjectGenerator()
    second = generate_project_summaries(
        _snapshot(), (_component(),), _config(), second_generator, cache
    )

    assert first_generator.calls == [
        "overview",
        "architecture",
        "testing_operations",
    ]
    assert second_generator.calls == []
    assert [result.run_report.cache_hit for result in first] == [False] * 3
    assert [result.run_report.cache_hit for result in second] == [True] * 3
    assert all(result.run_report.token_usage == TokenUsage() for result in second)


def test_changed_component_output_invalidates_every_project_dependency(
    tmp_path: Path,
) -> None:
    """A changed validated component hash changes every synthesis cache key."""

    cache = FilesystemArtifactCache(tmp_path / "cache")
    generate_project_summaries(
        _snapshot(), (_component(),), _config(), _ProjectGenerator(), cache
    )
    changed_generator = _ProjectGenerator()
    generate_project_summaries(
        _snapshot(),
        (_component("Changed component behavior."),),
        _config(),
        changed_generator,
        cache,
    )

    assert changed_generator.calls == [
        "overview",
        "architecture",
        "testing_operations",
    ]


def test_recoverable_project_responses_are_cached_and_reported(
    tmp_path: Path,
) -> None:
    """Every project kind retains all parser recoveries across cache reuse."""

    cache = FilesystemArtifactCache(tmp_path / "cache")
    first = generate_project_summaries(
        _snapshot(),
        (_component(),),
        _config(),
        _RecoveringProjectGenerator(),
        cache,
    )
    second_generator = _ProjectGenerator()
    second = generate_project_summaries(
        _snapshot(),
        (_component(),),
        _config(),
        second_generator,
        cache,
    )

    expected_codes = {
        "missing_section",
        "duplicate_section",
        "reordered_section",
        "unexpected_section",
        "unassigned_content",
        "invalid_evidence_path",
        "evidence_scope_fallback",
    }
    assert {artifact.artifact_kind for artifact in first} == {
        "overview",
        "architecture",
        "testing_operations",
    }
    for generated, cached in zip(first, second, strict=True):
        warnings = generated.artifact.generation_warnings
        assert {warning.code for warning in warnings} == expected_codes
        assert generated.run_report.generation_warning_count == len(warnings)
        assert cached.artifact.generation_warnings == warnings
        assert cached.run_report.generation_warning_count == len(warnings)
        assert cached.run_report.generation_warning_counts == (
            generated.run_report.generation_warning_counts
        )
        assert cached.run_report.cache_hit is True
    assert second_generator.calls == []


def test_empty_context_pattern_fails_before_any_provider_call(tmp_path: Path) -> None:
    """Every configured pattern must match the selected immutable snapshot."""

    generator = _ProjectGenerator()
    contexts = ProjectDocumentContextConfig(
        overview=("README.md", "missing.md"),
        architecture=("src/**/*.py",),
        testing_operations=("tests/**/*.py",),
    )

    with pytest.raises(ArtifactValidationError) as caught:
        generate_project_summaries(
            _snapshot(),
            (_component(),),
            _config(contexts=contexts),
            generator,
            FilesystemArtifactCache(tmp_path / "cache"),
        )

    assert caught.value.reason == "empty-context-pattern"
    assert generator.calls == []


def test_complete_synthesis_input_budget_fails_before_provider_call(
    tmp_path: Path,
) -> None:
    """Component JSON and source bytes are both included in hard input limits."""

    generator = _ProjectGenerator()
    with pytest.raises(ArtifactValidationError) as caught:
        generate_project_summaries(
            _snapshot(),
            (_component(),),
            _config(max_bytes=1),
            generator,
            FilesystemArtifactCache(tmp_path / "cache"),
        )

    assert caught.value.reason == "input-byte-limit"
    assert generator.calls == []
