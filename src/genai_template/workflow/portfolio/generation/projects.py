"""Cached project-document synthesis from validated component artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from time import perf_counter

from genai_template.workflow.portfolio.adapters.caches.file_cache import (
    CACHE_SCHEMA_VERSION,
)
from genai_template.workflow.portfolio.artifacts.fingerprints import (
    project_generation_fingerprint,
    sha256_canonical_json,
)
from genai_template.workflow.portfolio.config.models import GenerationConfig
from genai_template.workflow.portfolio.domain.components import ComponentSummaryArtifact
from genai_template.workflow.portfolio.domain.errors import ArtifactValidationError
from genai_template.workflow.portfolio.domain.generation import (
    ArtifactProvenance,
    CachedArtifact,
    GenerationRequest,
    TokenUsage,
)
from genai_template.workflow.portfolio.domain.projects import (
    ProjectArtifactKind,
    ProjectSummaryArtifact,
)
from genai_template.workflow.portfolio.domain.reports import ArtifactRunReport
from genai_template.workflow.portfolio.domain.snapshot import (
    RepositorySnapshot,
    SnapshotFile,
)
from genai_template.workflow.portfolio.domain.summaries import (
    OUTPUT_SCHEMA_VERSION,
    ArchitectureSummary,
    ProjectOverviewSummary,
    StructuredSummary,
    TestingOperationsSummary,
    summary_evidence_paths,
)
from genai_template.workflow.portfolio.generation.components import (
    _elapsed,
    _validate_provider_identity,
)
from genai_template.workflow.portfolio.generation.costs import estimate_generation_cost
from genai_template.workflow.portfolio.generation.prompts import (
    ARCHITECTURE_PROMPT,
    OVERVIEW_PROMPT,
    TESTING_OPERATIONS_PROMPT,
    PromptDefinition,
    assemble_project_prompt,
)
from genai_template.workflow.portfolio.generation.service import (
    generate_validated_summary,
)
from genai_template.workflow.portfolio.generation.validation import (
    validate_project_evidence,
)
from genai_template.workflow.portfolio.ports.artifact_cache import ArtifactCache
from genai_template.workflow.portfolio.ports.structured_generator import (
    StructuredSummaryGenerator,
)
from genai_template.workflow.portfolio.snapshot.selection import (
    matches_repository_pattern,
)


@dataclass(frozen=True)
class _ProjectSynthesisSpec:
    """Internal fully resolved inputs for one project-document call.

    Attributes:
        artifact_kind:
            Project artifact type to synthesize.
        prompt:
            Stable prompt definition for the artifact.
        output_type:
            Structured summary schema expected from the provider.
        patterns:
            Configured repository-context patterns.
        repository_files:
            Canonically ordered files matched by the patterns.
    """

    artifact_kind: ProjectArtifactKind
    prompt: PromptDefinition
    output_type: type[StructuredSummary]
    patterns: tuple[str, ...]
    repository_files: tuple[SnapshotFile, ...]


def generate_project_summaries(
    snapshot: RepositorySnapshot,
    components: tuple[ComponentSummaryArtifact, ...],
    generation: GenerationConfig,
    generator: StructuredSummaryGenerator,
    cache: ArtifactCache,
    *,
    clock: Callable[[], float] = perf_counter,
) -> tuple[ProjectSummaryArtifact, ...]:
    """Generate or reuse all balanced project summaries in fixed order.

    All context patterns and component artifacts are validated before any provider
    request. Each cache key depends on the exact repository context and ordered
    validated component artifact hashes.

    Args:
        snapshot:
            Canonical immutable repository snapshot.
        components:
            Validated component artifacts supplied to every synthesis call.
        generation:
            Validated balanced generation configuration.
        generator:
            Structured summary provider boundary.
        cache:
            Validated artifact cache boundary.
        clock:
            Monotonic clock used only for current-run latency reporting.

    Returns:
        Overview, architecture, and testing/operations artifacts in fixed order.

    Raises:
        ArtifactValidationError:
            If versions, component artifacts, context patterns, or input limits are
            invalid.
        ArtifactCacheError:
            If an existing cache entry is corrupt or cannot be persisted.
        EvidenceValidationError:
            If generated or cached evidence falls outside the synthesis scope.
        StructuredGenerationError:
            If provider generation or structured-output validation fails.
    """

    _validate_generation_versions(generation)
    component_outputs, component_hashes, component_evidence = _validate_components(
        snapshot, components
    )
    specs = _build_specs(snapshot, generation)
    return tuple(
        _generate_project_summary(
            snapshot,
            spec,
            component_outputs,
            component_hashes,
            component_evidence,
            generation,
            generator,
            cache,
            clock=clock,
        )
        for spec in specs
    )


def resolve_project_context(
    snapshot: RepositorySnapshot,
    patterns: tuple[str, ...],
    *,
    artifact_kind: ProjectArtifactKind,
) -> tuple[SnapshotFile, ...]:
    """Resolve repository patterns without silently accepting empty patterns.

    Args:
        snapshot:
            Canonical selected repository snapshot.
        patterns:
            Validated context patterns for one project document.
        artifact_kind:
            Project artifact being resolved, used in safe failure context.

    Returns:
        Unique matching snapshot files in canonical path order.

    Raises:
        ArtifactValidationError:
            If any configured pattern matches no selected snapshot file.
    """

    for pattern in patterns:
        if not any(
            matches_repository_pattern(file.path, pattern) for file in snapshot.files
        ):
            raise ArtifactValidationError(
                f"{artifact_kind} repository context pattern matches no files",
                generation_fingerprint="0" * 64,
                reason="empty-context-pattern",
            )
    return tuple(
        file
        for file in snapshot.files
        if any(matches_repository_pattern(file.path, pattern) for pattern in patterns)
    )


def _build_specs(
    snapshot: RepositorySnapshot,
    generation: GenerationConfig,
) -> tuple[_ProjectSynthesisSpec, ...]:
    """Build and preflight the three fixed project synthesis specifications.

    Args:
        snapshot:
            Canonical selected repository snapshot.
        generation:
            Balanced generation configuration.

    Returns:
        Fully resolved synthesis specifications in publication order.
    """

    contexts = generation.project_context
    definitions: tuple[
        tuple[
            ProjectArtifactKind,
            PromptDefinition,
            type[StructuredSummary],
            tuple[str, ...],
        ],
        ...,
    ] = (
        ("overview", OVERVIEW_PROMPT, ProjectOverviewSummary, contexts.overview),
        (
            "architecture",
            ARCHITECTURE_PROMPT,
            ArchitectureSummary,
            contexts.architecture,
        ),
        (
            "testing_operations",
            TESTING_OPERATIONS_PROMPT,
            TestingOperationsSummary,
            contexts.testing_operations,
        ),
    )
    return tuple(
        _ProjectSynthesisSpec(
            artifact_kind=kind,
            prompt=prompt,
            output_type=output_type,
            patterns=patterns,
            repository_files=resolve_project_context(
                snapshot, patterns, artifact_kind=kind
            ),
        )
        for kind, prompt, output_type, patterns in definitions
    )


def _validate_components(
    snapshot: RepositorySnapshot,
    components: tuple[ComponentSummaryArtifact, ...],
) -> tuple[dict[str, object], dict[str, str], tuple[str, ...]]:
    """Validate component identities and prepare deterministic synthesis inputs.

    Args:
        snapshot:
            Snapshot whose source identity must match every component.
        components:
            Component results supplied to project synthesis.

    Returns:
        Component outputs, artifact hashes, and combined evidence paths.

    Raises:
        ArtifactValidationError:
            If components are absent, duplicated, or internally inconsistent.
    """

    if not components:
        raise ArtifactValidationError(
            "project synthesis requires at least one component artifact",
            generation_fingerprint="0" * 64,
            reason="missing-components",
        )
    unit_ids = [component.unit_id for component in components]
    if len(unit_ids) != len(set(unit_ids)):
        raise ArtifactValidationError(
            "project synthesis component identifiers must be unique",
            generation_fingerprint="0" * 64,
            reason="duplicate-component-id",
        )

    outputs: dict[str, object] = {}
    hashes: dict[str, str] = {}
    evidence: set[str] = set()
    for component in sorted(components, key=lambda value: value.unit_id):
        artifact = component.artifact
        provenance = artifact.provenance
        expected_output = component.summary.model_dump(mode="json")
        if (
            provenance.artifact_kind != "component"
            or provenance.source_fingerprint != snapshot.source_fingerprint
            or artifact.structured_output != expected_output
            or artifact.output_hash != sha256_canonical_json(expected_output)
        ):
            raise ArtifactValidationError(
                "component artifact is inconsistent with project synthesis inputs",
                generation_fingerprint=provenance.generation_fingerprint,
                reason="invalid-component-artifact",
            )
        outputs[component.unit_id] = component.summary
        hashes[component.unit_id] = artifact.output_hash
        evidence.update(summary_evidence_paths(component.summary))
    return outputs, hashes, tuple(sorted(evidence))


def _generate_project_summary(
    snapshot: RepositorySnapshot,
    spec: _ProjectSynthesisSpec,
    component_outputs: Mapping[str, object],
    component_hashes: Mapping[str, str],
    component_evidence: tuple[str, ...],
    generation: GenerationConfig,
    generator: StructuredSummaryGenerator,
    cache: ArtifactCache,
    *,
    clock: Callable[[], float],
) -> ProjectSummaryArtifact:
    """Generate or reuse one validated project synthesis artifact.

    Args:
        snapshot:
            Exact immutable repository snapshot.
        spec:
            Resolved document inputs and expected output schema.
        component_outputs:
            Validated component summaries keyed by unit identifier.
        component_hashes:
            Validated component output hashes keyed by unit identifier.
        component_evidence:
            Combined evidence paths from supplied component summaries.
        generation:
            Balanced generation configuration.
        generator:
            Structured summary provider boundary.
        cache:
            Validated artifact cache boundary.
        clock:
            Monotonic current-run timing source.

    Returns:
        Validated project summary artifact and current-run metrics.
    """

    started_at = clock()
    context_identity = {
        "version": 1,
        "files": [
            {"path": file.path, "content_hash": file.content_hash}
            for file in spec.repository_files
        ],
    }
    fingerprint = project_generation_fingerprint(
        document_type=spec.artifact_kind,
        source_fingerprint=snapshot.source_fingerprint,
        repository_context_fingerprint=sha256_canonical_json(context_identity),
        component_artifact_hashes=component_hashes,
        prompt_id=spec.prompt.prompt_id,
        prompt_hash=spec.prompt.prompt_hash,
        output_schema_version=generation.output_schema_version,
        structured_generation=generation.structured_generation,
    )
    repository_paths = tuple(file.path for file in spec.repository_files)
    cached = cache.get(
        fingerprint,
        expected_kind=spec.artifact_kind,
        output_schema_version=generation.output_schema_version,
        output_type=spec.output_type,
    )
    if cached is not None:
        summary = spec.output_type.model_validate(cached.structured_output)
        validate_project_evidence(summary, repository_paths, component_evidence)
        return ProjectSummaryArtifact(
            artifact_kind=spec.artifact_kind,
            summary=summary,
            artifact=cached,
            run_report=_run_report(
                spec.artifact_kind,
                fingerprint,
                cache_hit=True,
                token_usage=TokenUsage(),
                estimated_cost=None,
                original_token_usage=cached.token_usage,
                original_estimated_cost=cached.estimated_cost,
                latency_seconds=_elapsed(clock, started_at),
            ),
        )

    _validate_input_limits(
        spec.repository_files,
        component_outputs,
        generation,
        fingerprint,
    )
    structured_generation = generation.structured_generation
    provenance = ArtifactProvenance(
        artifact_kind=spec.artifact_kind,
        generation_fingerprint=fingerprint,
        source_fingerprint=snapshot.source_fingerprint,
        component_artifact_hashes=tuple(
            component_hashes[unit_id] for unit_id in sorted(component_hashes)
        ),
        prompt_id=spec.prompt.prompt_id,
        prompt_hash=spec.prompt.prompt_hash,
        output_schema_version=generation.output_schema_version,
        provider=structured_generation.provider,
        model=structured_generation.model,
        temperature=structured_generation.inference.temperature,
        seed=structured_generation.inference.seed,
    )
    request = GenerationRequest(
        provenance=provenance,
        prompt=assemble_project_prompt(
            spec.prompt, spec.repository_files, component_outputs
        ),
        input_paths=repository_paths,
    )
    response = generate_validated_summary(
        generator,
        request,
        spec.output_type,
        repository_context_paths=repository_paths,
        component_evidence_paths=component_evidence,
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
    cache.put(artifact, output_type=spec.output_type)
    return ProjectSummaryArtifact(
        artifact_kind=spec.artifact_kind,
        summary=response.value,
        artifact=artifact,
        run_report=_run_report(
            spec.artifact_kind,
            fingerprint,
            cache_hit=False,
            token_usage=response.token_usage,
            estimated_cost=estimated_cost,
            original_token_usage=response.token_usage,
            original_estimated_cost=estimated_cost,
            latency_seconds=_elapsed(clock, started_at),
        ),
    )


def _validate_generation_versions(generation: GenerationConfig) -> None:
    """Require configuration to select implemented project prompts and schemas.

    Args:
        generation:
            Validated generation configuration.

    Raises:
        ArtifactValidationError:
            If a configured version has no implementation.
    """

    prompt_versions = {
        prompt.prompt_id.rsplit("-", maxsplit=1)[-1]
        for prompt in (
            OVERVIEW_PROMPT,
            ARCHITECTURE_PROMPT,
            TESTING_OPERATIONS_PROMPT,
        )
    }
    if generation.prompt_version not in prompt_versions or len(prompt_versions) != 1:
        raise ArtifactValidationError(
            "configured project prompt version is not implemented",
            generation_fingerprint="0" * 64,
            reason="unsupported-prompt-version",
        )
    if generation.output_schema_version != OUTPUT_SCHEMA_VERSION:
        raise ArtifactValidationError(
            "configured project output schema version is not implemented",
            generation_fingerprint="0" * 64,
            reason="unsupported-output-schema-version",
        )


def _validate_input_limits(
    repository_files: tuple[SnapshotFile, ...],
    component_outputs: Mapping[str, object],
    generation: GenerationConfig,
    fingerprint: str,
) -> None:
    """Fail before provider generation when complete synthesis inputs exceed limits.

    Args:
        repository_files:
            Exact direct repository context for the provider request.
        component_outputs:
            Complete component-summary mapping supplied to the request.
        generation:
            Configuration containing per-call limits.
        fingerprint:
            Stable identity of the attempted generation.

    Raises:
        ArtifactValidationError:
            If file count or complete source/artifact bytes exceed configured limits.
    """

    limits = generation.input_limits
    if len(repository_files) > limits.max_files:
        raise ArtifactValidationError(
            "project synthesis input exceeds the configured file limit",
            generation_fingerprint=fingerprint,
            reason="input-file-limit",
        )
    artifact_bytes = len(_component_input_bytes(component_outputs))
    source_bytes = sum(file.byte_size for file in repository_files)
    if source_bytes + artifact_bytes > limits.max_bytes:
        raise ArtifactValidationError(
            "project synthesis input exceeds the configured byte limit",
            generation_fingerprint=fingerprint,
            reason="input-byte-limit",
        )


def _component_input_bytes(component_outputs: Mapping[str, object]) -> bytes:
    """Serialize component synthesis inputs to their deterministic prompt bytes.

    Args:
        component_outputs:
            Validated component summaries keyed by unit identifier.

    Returns:
        Canonical UTF-8 JSON bytes supplied in the project prompt.
    """

    from genai_template.workflow.portfolio.artifacts.fingerprints import (
        canonical_json_bytes,
    )

    values: dict[str, object] = {}
    for unit_id, value in sorted(component_outputs.items()):
        model_dump = getattr(value, "model_dump", None)
        values[unit_id] = model_dump(mode="json") if callable(model_dump) else value
    return canonical_json_bytes(values)


def _run_report(
    artifact_kind: ProjectArtifactKind,
    fingerprint: str,
    *,
    cache_hit: bool,
    token_usage: TokenUsage,
    estimated_cost: Decimal | None,
    original_token_usage: TokenUsage,
    original_estimated_cost: Decimal | None,
    latency_seconds: float,
) -> ArtifactRunReport:
    """Build one current-run project artifact report.

    Args:
        artifact_kind:
            Project document type.
        fingerprint:
            Stable generation fingerprint.
        cache_hit:
            Whether this run reused an artifact.
        token_usage:
            Tokens billed during this run.
        estimated_cost:
            Estimated cost billed during this run.
        original_token_usage:
            Usage recorded by the original provider call.
        original_estimated_cost:
            Cost recorded by the original provider call.
        latency_seconds:
            Current-run operation duration.

    Returns:
        Immutable artifact run report.
    """

    return ArtifactRunReport(
        artifact_kind=artifact_kind,
        generation_fingerprint=fingerprint,
        cache_hit=cache_hit,
        token_usage=token_usage,
        estimated_cost=estimated_cost,
        original_token_usage=original_token_usage,
        original_estimated_cost=original_estimated_cost,
        latency_seconds=latency_seconds,
    )
