"""Tests for cached deterministic component-summary generation."""

from __future__ import annotations

import hashlib
from decimal import Decimal
from pathlib import Path

import pytest

from genai_template.config import settings
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
    GenerationRequest,
    ProviderAuditMetadata,
    SnapshotFile,
    SummaryPlan,
    SummaryUnitPlan,
    TextGenerationError,
    TextGenerationResponse,
    TokenUsage,
)
from genai_template.workflow.portfolio.generation import generate_component_summaries
from genai_template.workflow.portfolio.generation.prompts import PromptDefinition
from genai_template.workflow.portfolio.infrastructure.caches import (
    FilesystemArtifactCache,
)


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

    def __init__(
        self,
        *,
        fail: bool = False,
        model: str = "test-model",
        evidence_path: str | None = None,
        response_text: str | None = None,
    ) -> None:
        """Initialize deterministic fake behavior.

        Args:
            fail:
                Whether calls should raise a provider-neutral failure.
            model:
                Provider model identity to return.
            evidence_path:
                Optional evidence path replacing each requested path.
            response_text:
                Optional complete response replacing canonical Markdown.
        """

        self.calls: list[tuple[str, ...]] = []
        self.fail = fail
        self.model = model
        self.evidence_path = evidence_path
        self.response_text = response_text

    def generate(self, request: GenerationRequest) -> TextGenerationResponse:
        """Return path-scoped Markdown.

        Args:
            request:
                Component generation request.
        Returns:
            Deterministic plain-text provider response.

        Raises:
            TextGenerationError:
                When configured to simulate failure.
        """

        paths = request.input_paths
        self.calls.append(paths)
        if self.fail:
            raise TextGenerationError(
                "simulated generation failure",
                provider="ollama",
                model="test-model",
                reason="provider-failure",
            )
        headings = (
            "Responsibilities",
            "Important Abstractions",
            "Behavior",
            "Constraints",
            "Testing Evidence",
        )
        evidence_path = self.evidence_path or paths[0]
        text = self.response_text or "\n\n".join(
            f"## {heading}\n\nSummary for {paths[0]}.\n\n"
            f"### Evidence\n\n- {evidence_path}"
            for heading in headings
        )
        return TextGenerationResponse(
            text=text,
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


def _plan(
    second_fingerprint: str = "4" * 64,
    *,
    source_fingerprint: str = "b" * 64,
) -> SummaryPlan:
    """Build a two-unit plan in explicit configuration order.

    Args:
        second_fingerprint:
            Input identity for the second unit.
        source_fingerprint:
            Identity of the complete selected source snapshot.

    Returns:
        Deterministic summary plan.
    """

    return SummaryPlan(
        project_slug="sample",
        resolved_commit_sha="a" * 40,
        source_fingerprint=source_fingerprint,
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


def _config(
    *,
    model: str = "test-model",
    temperature: float = 0,
    seed: int | None = 7,
) -> GenerationConfig:
    """Build valid balanced generation settings.

    Args:
        model:
            Structured-generation model name.
        temperature:
            Content-affecting sampling temperature.
        seed:
            Optional content-affecting sampling seed.

    Returns:
        Valid generation configuration.
    """

    return GenerationConfig(
        profile="balanced",
        structured_generation=StructuredGenerationConfig(
            provider="ollama",
            model=model,
            inference=InferenceConfig(temperature=temperature, seed=seed),
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
    assert first[0].artifact.original_token_usage.input_tokens == 100
    assert first[0].artifact.original_estimated_cost == Decimal("0.0004")
    expected_response = "\n\n".join(
        f"## {heading}\n\nSummary for src/api.py.\n\n" "### Evidence\n\n- src/api.py"
        for heading in (
            "Responsibilities",
            "Important Abstractions",
            "Behavior",
            "Constraints",
            "Testing Evidence",
        )
    )
    assert (
        first[0].artifact.raw_response_hash
        == hashlib.sha256(expected_response.encode("utf-8")).hexdigest()
    )
    assert second[0].artifact.raw_response_hash == first[0].artifact.raw_response_hash
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


def test_source_prompt_and_inference_changes_invalidate_all_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every global content-affecting identity change causes fresh calls."""

    cache = FilesystemArtifactCache(tmp_path / "cache")
    generate_component_summaries(_plan(), _config(), _CountingGenerator(), cache)

    source_change = _CountingGenerator()
    generate_component_summaries(
        _plan(source_fingerprint="c" * 64),
        _config(),
        source_change,
        cache,
    )
    inference_change = _CountingGenerator()
    generate_component_summaries(
        _plan(source_fingerprint="c" * 64),
        _config(temperature=0.2, seed=11),
        inference_change,
        cache,
    )
    monkeypatch.setattr(
        "genai_template.workflow.portfolio.generation.components.COMPONENT_PROMPT",
        PromptDefinition(
            prompt_id="portfolio-component-v1",
            artifact_kind="component",
            instructions="Changed component summary instructions.",
        ),
    )
    prompt_change = _CountingGenerator()
    generate_component_summaries(
        _plan(source_fingerprint="c" * 64),
        _config(temperature=0.2, seed=11),
        prompt_change,
        cache,
    )

    expected_calls = [("src/api.py",), ("tests/test_api.py",)]
    assert source_change.calls == expected_calls
    assert inference_change.calls == expected_calls
    assert prompt_change.calls == expected_calls


def test_invalid_evidence_warnings_and_fallback_are_reused_from_cache(
    tmp_path: Path,
) -> None:
    """Filtered evidence keeps usable content and its original audit warnings."""

    cache = FilesystemArtifactCache(tmp_path / "cache")
    first_generator = _CountingGenerator(evidence_path="outside/scope.py")
    first = generate_component_summaries(_plan(), _config(), first_generator, cache)
    second_generator = _CountingGenerator()
    second = generate_component_summaries(_plan(), _config(), second_generator, cache)

    first_component = first[0]
    warning_codes = tuple(
        warning.code for warning in first_component.artifact.generation_warnings
    )
    assert first_component.summary.responsibilities.content == (
        "Summary for src/api.py.",
    )
    assert first_component.summary.responsibilities.evidence_paths == ("src/api.py",)
    assert warning_codes.count("invalid_evidence_path") == 5
    assert warning_codes.count("evidence_scope_fallback") == 5
    assert second_generator.calls == []
    assert second[0].artifact.generation_warnings == (
        first_component.artifact.generation_warnings
    )


def test_empty_response_is_not_cached(tmp_path: Path) -> None:
    """Whitespace-only provider output cannot create a reusable entry."""

    cache_root = tmp_path / "cache"
    with pytest.raises(ValueError, match="must be non-empty"):
        generate_component_summaries(
            _plan(),
            _config(),
            _CountingGenerator(response_text="  "),
            FilesystemArtifactCache(cache_root),
        )

    assert not cache_root.exists() or list(cache_root.iterdir()) == []


def test_generation_failure_is_not_cached(tmp_path: Path) -> None:
    """A failure before validation leaves no reusable cache entry."""

    cache_root = tmp_path / "cache"
    with pytest.raises(TextGenerationError):
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
