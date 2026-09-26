"""Tests for cached deterministic component-summary generation."""

from __future__ import annotations

import hashlib
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest

from genai_template.config import settings
from genai_template.workflow.portfolio.adapters.caches import (
    FilesystemArtifactCache,
)
from genai_template.workflow.portfolio.config import (
    GenerationConfig,
    GenerationInputLimits,
    GenerationLocations,
    InferenceConfig,
    ProjectDocumentContextConfig,
    StructuredGenerationConfig,
    TokenPricingConfig,
)
from genai_template.workflow.portfolio.domain import (
    ArtifactValidationError,
    ComponentSummary,
    EvidenceSection,
    GenerationRequest,
    ProviderAuditMetadata,
    SnapshotFile,
    StructuredGenerationError,
    StructuredSummary,
    SummaryPlan,
    SummaryUnitPlan,
    TokenUsage,
)
from genai_template.workflow.portfolio.generation import generate_component_summaries
from genai_template.workflow.portfolio.ports import StructuredGenerationResponse


class _CountingGenerator:
    """Deterministic structured generator that records provider calls.

    Attributes:
        calls:
            Unit input-path tuples received in call order.
        fail:
            Whether to simulate structured generation failure.
        model:
            Provider model identity returned by the fake.
    """

    def __init__(self, *, fail: bool = False, model: str = "test-model") -> None:
        """Initialize deterministic fake behavior.

        Args:
            fail:
                Whether calls should raise a provider-neutral failure.
            model:
                Provider model identity to return.
        """

        self.calls: list[tuple[str, ...]] = []
        self.fail = fail
        self.model = model

    def generate[SummaryT: StructuredSummary](
        self, request: GenerationRequest, output_type: type[SummaryT]
    ) -> StructuredGenerationResponse[SummaryT]:
        """Return a path-scoped valid summary.

        Args:
            request:
                Component generation request.
            output_type:
                Expected component output type.

        Returns:
            Deterministic validated provider response.

        Raises:
            StructuredGenerationError:
                When configured to simulate failure.
        """

        paths = request.input_paths
        self.calls.append(paths)
        if self.fail:
            raise StructuredGenerationError(
                "simulated generation failure",
                provider="ollama",
                model="test-model",
                reason="provider-failure",
            )
        section = EvidenceSection(
            content=(f"Summary for {paths[0]}.",), evidence_paths=(paths[0],)
        )
        component = ComponentSummary(
            responsibilities=section,
            important_abstractions=section,
            behavior=section,
            constraints=section,
            testing_evidence=section,
        )
        value = cast(
            SummaryT,
            output_type.model_validate(component.model_dump(mode="json")),
        )
        return StructuredGenerationResponse[SummaryT](
            value=value,
            provider="ollama",
            model=self.model,
            token_usage=TokenUsage(input_tokens=100, output_tokens=25),
            provider_metadata=ProviderAuditMetadata(request_id="request-1"),
        )


def _file(path: str, text: str) -> SnapshotFile:
    """Build one canonical snapshot file.

    Args:
        path:
            Repository-relative path.
        text:
            Normalized source content.

    Returns:
        Canonical snapshot file.
    """

    content = text.encode("utf-8")
    return SnapshotFile(
        path=path,
        text=text,
        content_hash=hashlib.sha256(content).hexdigest(),
        byte_size=len(content),
    )


def _plan(second_fingerprint: str = "4" * 64) -> SummaryPlan:
    """Build a two-unit plan in explicit configuration order.

    Args:
        second_fingerprint:
            Input identity for the second unit.

    Returns:
        Deterministic summary plan.
    """

    return SummaryPlan(
        project_slug="sample",
        resolved_commit_sha="a" * 40,
        source_fingerprint="b" * 64,
        units=(
            SummaryUnitPlan(
                unit_id="api",
                input_fingerprint="3" * 64,
                files=(_file("src/api.py", "api source"),),
            ),
            SummaryUnitPlan(
                unit_id="tests",
                input_fingerprint=second_fingerprint,
                files=(_file("tests/test_api.py", "test source"),),
            ),
        ),
    )


def _config(*, model: str = "test-model") -> GenerationConfig:
    """Build valid balanced generation settings.

    Args:
        model:
            Structured-generation model name.

    Returns:
        Valid generation configuration.
    """

    return GenerationConfig(
        profile="balanced",
        structured_generation=StructuredGenerationConfig(
            provider="ollama",
            model=model,
            inference=InferenceConfig(temperature=0, seed=7),
        ),
        prompt_version="v1",
        output_schema_version="v1",
        project_context=ProjectDocumentContextConfig(
            overview=("README.md",),
            architecture=("src/**/*.py",),
            testing_operations=("tests/**/*.py",),
        ),
        input_limits=GenerationInputLimits(max_files=10, max_bytes=10_000),
        locations=GenerationLocations(
            cache=settings.REPO_ROOT / "storage/test-portfolio-cache",
            publication=settings.REPO_ROOT / "data/test-portfolio",
        ),
        pricing=TokenPricingConfig(
            input_per_million_tokens=Decimal(2),
            output_per_million_tokens=Decimal(8),
        ),
    )


def test_second_unchanged_run_uses_cache_without_generator_calls(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Components are reused without persisting or logging generation inputs."""

    cache = FilesystemArtifactCache(tmp_path / "cache")
    first_generator = _CountingGenerator()
    first = generate_component_summaries(_plan(), _config(), first_generator, cache)
    second_generator = _CountingGenerator()
    second = generate_component_summaries(_plan(), _config(), second_generator, cache)

    assert first_generator.calls == [("src/api.py",), ("tests/test_api.py",)]
    assert second_generator.calls == []
    assert [result.unit_id for result in second] == ["api", "tests"]
    assert [result.run_report.cache_hit for result in first] == [False, False]
    assert [result.run_report.cache_hit for result in second] == [True, True]
    assert second[0].run_report.token_usage == TokenUsage()
    assert first[0].artifact.token_usage.input_tokens == 100
    assert first[0].artifact.estimated_cost == Decimal("0.0004")
    cache_bytes = b"".join(path.read_bytes() for path in cache.root.iterdir())
    assert b"api source" not in cache_bytes
    assert b"test source" not in cache_bytes
    assert str(tmp_path).encode() not in cache_bytes
    assert "api source" not in caplog.text
    assert "test source" not in caplog.text


def test_selective_unit_and_global_model_invalidation(tmp_path: Path) -> None:
    """Unit identity changes are selective while model changes affect all keys."""

    cache = FilesystemArtifactCache(tmp_path / "cache")
    generate_component_summaries(_plan(), _config(), _CountingGenerator(), cache)

    selective = _CountingGenerator()
    generate_component_summaries(
        _plan(second_fingerprint="5" * 64), _config(), selective, cache
    )
    global_change = _CountingGenerator(model="another-model")
    generate_component_summaries(
        _plan(second_fingerprint="5" * 64),
        _config(model="another-model"),
        global_change,
        cache,
    )

    assert selective.calls == [("tests/test_api.py",)]
    assert global_change.calls == [("src/api.py",), ("tests/test_api.py",)]


def test_generation_failure_is_not_cached(tmp_path: Path) -> None:
    """A failure before validation leaves no reusable cache entry."""

    cache_root = tmp_path / "cache"
    with pytest.raises(StructuredGenerationError):
        generate_component_summaries(
            _plan(),
            _config(),
            _CountingGenerator(fail=True),
            FilesystemArtifactCache(cache_root),
        )

    assert not cache_root.exists() or list(cache_root.iterdir()) == []


def test_provider_identity_mismatch_is_not_cached(tmp_path: Path) -> None:
    """Provider provenance is checked before validated output is persisted."""

    cache_root = tmp_path / "cache"
    with pytest.raises(ArtifactValidationError) as caught:
        generate_component_summaries(
            _plan(),
            _config(model="configured-model"),
            _CountingGenerator(),
            FilesystemArtifactCache(cache_root),
        )

    assert caught.value.reason == "provider-identity-mismatch"
    assert not cache_root.exists() or list(cache_root.iterdir()) == []
