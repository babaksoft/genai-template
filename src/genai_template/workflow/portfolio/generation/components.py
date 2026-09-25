"""Cached component-summary generation in deterministic plan order."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from genai_template.workflow.portfolio.artifacts.cache import CACHE_SCHEMA_VERSION
from genai_template.workflow.portfolio.artifacts.fingerprints import (
    component_generation_fingerprint,
    sha256_canonical_json,
)
from genai_template.workflow.portfolio.config.models import GenerationConfig
from genai_template.workflow.portfolio.domain.components import (
    ComponentSummaryArtifact,
)
from genai_template.workflow.portfolio.domain.errors import ArtifactValidationError
from genai_template.workflow.portfolio.domain.generation import (
    ArtifactProvenance,
    CachedArtifact,
    GenerationRequest,
    TokenUsage,
)
from genai_template.workflow.portfolio.domain.reports import ArtifactRunReport
from genai_template.workflow.portfolio.domain.summaries import (
    OUTPUT_SCHEMA_VERSION,
    ComponentSummary,
)
from genai_template.workflow.portfolio.domain.summary import (
    SummaryPlan,
    SummaryUnitPlan,
)
from genai_template.workflow.portfolio.generation.costs import (
    estimate_generation_cost,
)
from genai_template.workflow.portfolio.generation.prompts import (
    COMPONENT_PROMPT,
    assemble_component_prompt,
)
from genai_template.workflow.portfolio.generation.service import (
    generate_validated_summary,
)
from genai_template.workflow.portfolio.generation.validation import (
    validate_component_evidence,
)
from genai_template.workflow.portfolio.ports.artifact_cache import ArtifactCache
from genai_template.workflow.portfolio.ports.structured_generator import (
    StructuredSummaryGenerator,
)


def generate_component_summaries(
    plan: SummaryPlan,
    generation: GenerationConfig,
    generator: StructuredSummaryGenerator,
    cache: ArtifactCache,
    *,
    clock: Callable[[], float] = perf_counter,
) -> tuple[ComponentSummaryArtifact, ...]:
    """Generate or reuse every configured component in plan order.

    Args:
        plan:
            Deterministic logical summary plan.
        generation:
            Validated balanced generation configuration.
        generator:
            Structured summary provider boundary.
        cache:
            Validated artifact cache boundary.
        clock:
            Monotonic clock used only for current-run latency reporting.

    Returns:
        Component results in the exact order of the supplied plan.

    Raises:
        ArtifactCacheError:
            If a present cache entry is corrupt or cannot be persisted.
        ArtifactValidationError:
            If configuration or provider identity does not match the request.
        EvidenceValidationError:
            If generated or cached evidence falls outside a component's files.
        StructuredGenerationError:
            If the provider call or structured output validation fails.
    """

    _validate_generation_versions(generation)
    return tuple(
        _generate_component_summary(
            plan,
            unit,
            generation,
            generator,
            cache,
            clock=clock,
        )
        for unit in plan.units
    )


def _generate_component_summary(
    plan: SummaryPlan,
    unit: SummaryUnitPlan,
    generation: GenerationConfig,
    generator: StructuredSummaryGenerator,
    cache: ArtifactCache,
    *,
    clock: Callable[[], float],
) -> ComponentSummaryArtifact:
    """Generate or reuse one validated component artifact.

    Args:
        plan:
            Parent summary plan carrying the source identity.
        unit:
            Exact logical unit and repository files to summarize.
        generation:
            Validated balanced generation configuration.
        generator:
            Structured summary provider boundary.
        cache:
            Validated artifact cache boundary.
        clock:
            Monotonic current-run timing source.

    Returns:
        Validated component artifact and current-run metrics.
    """

    started_at = clock()
    structured_generation = generation.structured_generation
    fingerprint = component_generation_fingerprint(
        source_fingerprint=plan.source_fingerprint,
        unit_id=unit.unit_id,
        unit_input_fingerprint=unit.input_fingerprint,
        prompt_id=COMPONENT_PROMPT.prompt_id,
        prompt_hash=COMPONENT_PROMPT.prompt_hash,
        output_schema_version=generation.output_schema_version,
        structured_generation=structured_generation,
    )
    cached = cache.get(
        fingerprint,
        expected_kind="component",
        output_schema_version=generation.output_schema_version,
        output_type=ComponentSummary,
    )
    paths = tuple(file.path for file in unit.files)
    if cached is not None:
        summary = ComponentSummary.model_validate(cached.structured_output)
        validate_component_evidence(summary, paths)
        return ComponentSummaryArtifact(
            unit_id=unit.unit_id,
            summary=summary,
            artifact=cached,
            run_report=ArtifactRunReport(
                artifact_kind="component",
                generation_fingerprint=fingerprint,
                cache_hit=True,
                token_usage=TokenUsage(),
                estimated_cost=None,
                latency_seconds=_elapsed(clock, started_at),
            ),
        )

    _validate_input_limits(unit, generation, fingerprint)
    provenance = ArtifactProvenance(
        artifact_kind="component",
        generation_fingerprint=fingerprint,
        source_fingerprint=plan.source_fingerprint,
        unit_input_fingerprint=unit.input_fingerprint,
        prompt_id=COMPONENT_PROMPT.prompt_id,
        prompt_hash=COMPONENT_PROMPT.prompt_hash,
        output_schema_version=generation.output_schema_version,
        provider=structured_generation.provider,
        model=structured_generation.model,
        temperature=structured_generation.inference.temperature,
        seed=structured_generation.inference.seed,
    )
    request = GenerationRequest(
        provenance=provenance,
        prompt=assemble_component_prompt(COMPONENT_PROMPT, unit.files),
        input_paths=paths,
    )
    response = generate_validated_summary(
        generator,
        request,
        ComponentSummary,
        repository_context_paths=paths,
    )
    _validate_provider_identity(
        response.provider,
        response.model,
        generation,
        fingerprint,
    )
    structured_output = response.value.model_dump(mode="json")
    estimated_cost = estimate_generation_cost(response.token_usage, generation.pricing)
    artifact = CachedArtifact(
        cache_schema_version=CACHE_SCHEMA_VERSION,
        provenance=provenance,
        structured_output=structured_output,
        output_hash=sha256_canonical_json(structured_output),
        token_usage=response.token_usage,
        estimated_cost=estimated_cost,
        provider_metadata=response.provider_metadata,
    )
    cache.put(artifact, output_type=ComponentSummary)
    return ComponentSummaryArtifact(
        unit_id=unit.unit_id,
        summary=response.value,
        artifact=artifact,
        run_report=ArtifactRunReport(
            artifact_kind="component",
            generation_fingerprint=fingerprint,
            cache_hit=False,
            token_usage=response.token_usage,
            estimated_cost=estimated_cost,
            latency_seconds=_elapsed(clock, started_at),
        ),
    )


def _validate_generation_versions(generation: GenerationConfig) -> None:
    """Require configuration to select the implemented prompt and schema.

    Args:
        generation:
            Validated generation configuration.

    Raises:
        ArtifactValidationError:
            If the configured version has no implementation.
    """

    fingerprint = "0" * 64
    prompt_version = COMPONENT_PROMPT.prompt_id.rsplit("-", maxsplit=1)[-1]
    if generation.prompt_version != prompt_version:
        raise ArtifactValidationError(
            "configured component prompt version is not implemented",
            generation_fingerprint=fingerprint,
            reason="unsupported-prompt-version",
        )
    if generation.output_schema_version != OUTPUT_SCHEMA_VERSION:
        raise ArtifactValidationError(
            "configured component output schema version is not implemented",
            generation_fingerprint=fingerprint,
            reason="unsupported-output-schema-version",
        )


def _validate_input_limits(
    unit: SummaryUnitPlan,
    generation: GenerationConfig,
    fingerprint: str,
) -> None:
    """Fail before provider generation when component inputs exceed limits.

    Args:
        unit:
            Component unit whose source input will be sent.
        generation:
            Generation configuration containing per-call limits.
        fingerprint:
            Stable identity of the attempted generation.

    Raises:
        ArtifactValidationError:
            If file count or normalized source bytes exceed configured limits.
    """

    limits = generation.input_limits
    if len(unit.files) > limits.max_files:
        raise ArtifactValidationError(
            "component input exceeds the configured file limit",
            generation_fingerprint=fingerprint,
            reason="input-file-limit",
        )
    if sum(file.byte_size for file in unit.files) > limits.max_bytes:
        raise ArtifactValidationError(
            "component input exceeds the configured byte limit",
            generation_fingerprint=fingerprint,
            reason="input-byte-limit",
        )


def _validate_provider_identity(
    provider: str,
    model: str,
    generation: GenerationConfig,
    fingerprint: str,
) -> None:
    """Ensure provider accounting belongs to the requested provenance.

    Args:
        provider:
            Provider identity returned by the adapter.
        model:
            Model identity returned by the adapter.
        generation:
            Generation configuration used for the request.
        fingerprint:
            Stable identity of the attempted generation.

    Raises:
        ArtifactValidationError:
            If the response identity differs from requested provenance.
    """

    expected = generation.structured_generation
    if provider != expected.provider or model != expected.model:
        raise ArtifactValidationError(
            "structured generation response provenance does not match its request",
            generation_fingerprint=fingerprint,
            reason="provider-identity-mismatch",
        )


def _elapsed(clock: Callable[[], float], started_at: float) -> float:
    """Calculate a non-negative current-run elapsed duration.

    Args:
        clock:
            Monotonic timing source.
        started_at:
            Timing value captured at operation start.

    Returns:
        Non-negative elapsed seconds.
    """

    return max(0.0, clock() - started_at)
