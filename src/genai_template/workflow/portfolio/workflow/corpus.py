"""Complete LlamaIndex workflow for one Portfolio corpus generation run."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from decimal import Decimal
from pathlib import Path
from time import perf_counter

from llama_index.core.workflow import StopEvent, Workflow, step

from genai_template.observability import application_span
from genai_template.workflow.portfolio.adapters.caches.file_cache import (
    FilesystemArtifactCache,
)
from genai_template.workflow.portfolio.config.loader import load_portfolio_config
from genai_template.workflow.portfolio.config.models import (
    GenerationConfig,
    PortfolioConfig,
)
from genai_template.workflow.portfolio.corpus.manifest import (
    build_corpus_manifest,
    generation_configuration_fingerprint,
)
from genai_template.workflow.portfolio.corpus.publisher import (
    PublicationResult,
    publish_corpus,
)
from genai_template.workflow.portfolio.corpus.renderers import (
    render_balanced_documents,
)
from genai_template.workflow.portfolio.corpus.validation import (
    validate_rendered_corpus,
)
from genai_template.workflow.portfolio.domain.generation import (
    GenerationRequest,
    TokenUsage,
)
from genai_template.workflow.portfolio.domain.reports import (
    ArtifactRunReport,
    GenerationRunReport,
    StepRunReport,
)
from genai_template.workflow.portfolio.domain.summaries import StructuredSummary
from genai_template.workflow.portfolio.generation.components import (
    generate_component_summaries,
)
from genai_template.workflow.portfolio.generation.projects import (
    generate_project_summaries,
)
from genai_template.workflow.portfolio.ports.artifact_cache import ArtifactCache
from genai_template.workflow.portfolio.ports.repository import RepositoryReader
from genai_template.workflow.portfolio.ports.structured_generator import (
    StructuredGenerationResponse,
    StructuredSummaryGenerator,
)
from genai_template.workflow.portfolio.snapshot.planning import build_summary_plan
from genai_template.workflow.portfolio.snapshot.selection import (
    build_repository_snapshot,
)
from genai_template.workflow.portfolio.workflow.events import (
    ComponentsGeneratedEvent,
    ConfigurationSelectedEvent,
    CorpusValidatedEvent,
    DocumentsRenderedEvent,
    GenerateCorpusStartEvent,
    ProjectsGeneratedEvent,
    SnapshotSelectedEvent,
    SummaryPlannedEvent,
)

ConfigLoader = Callable[[Path], PortfolioConfig]
GeneratorFactory = Callable[[GenerationConfig], StructuredSummaryGenerator]
CacheFactory = Callable[[Path], ArtifactCache]
CorpusPublisher = Callable[..., PublicationResult]


class _ObservedGenerator:
    """Add content-safe application spans around actual provider calls.

    Attributes:
        delegate:
            Injected structured-generation implementation.
    """

    def __init__(self, delegate: StructuredSummaryGenerator) -> None:
        """Initialize the observed provider boundary.

        Args:
            delegate:
                Structured-generation implementation to invoke.
        """

        self.delegate = delegate

    def generate[SummaryT: StructuredSummary](
        self,
        request: GenerationRequest,
        output_type: type[SummaryT],
    ) -> StructuredGenerationResponse[SummaryT]:
        """Generate one cache-miss artifact under a redacted provider span.

        Args:
            request:
                Fully identified request; its prompt is never attached to the span.
            output_type:
                Strict structured output model.

        Returns:
            Validated delegate response.
        """

        provenance = request.provenance
        attributes = {
            "artifact.kind": provenance.artifact_kind,
            "generation.fingerprint": provenance.generation_fingerprint,
            "source.fingerprint": provenance.source_fingerprint,
            "generation.provider": provenance.provider,
            "generation.model": provenance.model,
            "cache.hit": False,
            "input.path_count": len(request.input_paths),
        }
        with application_span("portfolio.provider", "LLM", attributes):
            return self.delegate.generate(request, output_type)


class PortfolioCorpusWorkflow(Workflow):
    """Run the eight sequential corpus lifecycle steps.

    Attributes:
        repository_reader:
            Injected immutable repository reader.
        generator_factory:
            Injected structured-generation adapter factory.
        config_loader:
            Injected validated configuration loader.
        cache_factory:
            Injected validated artifact-cache factory.
        publisher:
            Injected atomic publication boundary.
        clock:
            Injected monotonic timing source.
    """

    def __init__(
        self,
        *,
        repository_reader: RepositoryReader,
        generator_factory: GeneratorFactory,
        config_loader: ConfigLoader = load_portfolio_config,
        cache_factory: CacheFactory = FilesystemArtifactCache,
        publisher: CorpusPublisher = publish_corpus,
        clock: Callable[[], float] = perf_counter,
        timeout: float | None = 600.0,
    ) -> None:
        """Initialize injected workflow boundaries.

        Args:
            repository_reader:
                Reader for the configured immutable repository revision.
            generator_factory:
                Factory for the configured structured-generation adapter.
            config_loader:
                Validated Portfolio configuration loader.
            cache_factory:
                Factory for the artifact cache selected by configuration.
            publisher:
                Atomic corpus publication boundary.
            clock:
                Monotonic clock used for volatile run metrics.
            timeout:
                Maximum LlamaIndex workflow duration in seconds.
        """

        super().__init__(timeout=timeout)
        self.repository_reader = repository_reader
        self.generator_factory = generator_factory
        self.config_loader = config_loader
        self.cache_factory = cache_factory
        self.publisher = publisher
        self.clock = clock

    @step
    async def select_configuration(
        self, event: GenerateCorpusStartEvent
    ) -> ConfigurationSelectedEvent:
        """Load generation settings and select exactly one project.

        Args:
            event:
                Workflow request with configuration path and project slug.

        Returns:
            Selected immutable configuration values.

        Raises:
            ValueError:
                If the project is absent or generation settings are missing.
        """

        run_started_at = self.clock()
        step_started_at = self.clock()
        with application_span(
            "portfolio.configuration", "CHAIN", {"project.id": event.project_slug}
        ):
            config = self.config_loader(event.config_path)
            matches = tuple(
                project
                for project in config.projects
                if project.slug == event.project_slug
            )
            if len(matches) != 1:
                raise ValueError(f"configured project not found: {event.project_slug}")

            generation = config.require_generation()

        return ConfigurationSelectedEvent(
            project=matches[0],
            generation=generation,
            run_started_at=run_started_at,
            steps=(_step_report("configuration", self.clock, step_started_at),),
        )

    @step
    async def select_snapshot(
        self, event: ConfigurationSelectedEvent
    ) -> SnapshotSelectedEvent:
        """Read a committed ref and select its canonical snapshot.

        Args:
            event:
                Selected project and generation configuration.

        Returns:
            Canonical immutable repository snapshot.
        """

        started_at = self.clock()
        with application_span(
            "portfolio.snapshot", "RETRIEVER", {"project.id": event.project.slug}
        ):
            read_result = self.repository_reader.read(event.project.repository)
            snapshot = build_repository_snapshot(
                event.project.slug, read_result, event.project.selection
            )

        return SnapshotSelectedEvent(
            project=event.project,
            generation=event.generation,
            snapshot=snapshot,
            run_started_at=event.run_started_at,
            steps=_append_step(event.steps, "snapshot", self.clock, started_at),
        )

    @step
    async def plan_summaries(self, event: SnapshotSelectedEvent) -> SummaryPlannedEvent:
        """Build the deterministic configured logical summary plan.

        Args:
            event:
                Selected snapshot and configuration.

        Returns:
            Complete logical summary plan.
        """

        started_at = self.clock()
        with application_span(
            "portfolio.planning",
            "CHAIN",
            {"source.fingerprint": event.snapshot.source_fingerprint},
        ):
            plan = build_summary_plan(event.snapshot, event.project.summary_units)

        return SummaryPlannedEvent(
            project=event.project,
            generation=event.generation,
            snapshot=event.snapshot,
            plan=plan,
            run_started_at=event.run_started_at,
            steps=_append_step(event.steps, "planning", self.clock, started_at),
        )

    @step
    async def generate_components(
        self, event: SummaryPlannedEvent
    ) -> ComponentsGeneratedEvent:
        """Generate or reuse component artifacts sequentially.

        Args:
            event:
                Planned component work and generation settings.

        Returns:
            Validated component artifacts.
        """

        started_at = self.clock()
        generator = _ObservedGenerator(self.generator_factory(event.generation))
        cache = self.cache_factory(event.generation.locations.cache)
        with application_span(
            "portfolio.components", "CHAIN", {"artifact.count": len(event.plan.units)}
        ):
            components = generate_component_summaries(
                event.plan,
                event.generation,
                generator,
                cache,
                clock=self.clock,
            )

        return ComponentsGeneratedEvent(
            project=event.project,
            generation=event.generation,
            snapshot=event.snapshot,
            components=components,
            run_started_at=event.run_started_at,
            steps=_append_step(event.steps, "components", self.clock, started_at),
        )

    @step
    async def generate_projects(
        self, event: ComponentsGeneratedEvent
    ) -> ProjectsGeneratedEvent:
        """Generate or reuse the three project synthesis artifacts.

        Args:
            event:
                Validated component artifacts and snapshot.

        Returns:
            Component and project artifacts.
        """

        started_at = self.clock()
        generator = _ObservedGenerator(self.generator_factory(event.generation))
        cache = self.cache_factory(event.generation.locations.cache)
        with application_span(
            "portfolio.project_synthesis",
            "CHAIN",
            {"component.count": len(event.components)},
        ):
            projects = generate_project_summaries(
                event.snapshot,
                event.components,
                event.generation,
                generator,
                cache,
                clock=self.clock,
            )
        return ProjectsGeneratedEvent(
            project=event.project,
            generation=event.generation,
            snapshot=event.snapshot,
            components=event.components,
            projects=projects,
            run_started_at=event.run_started_at,
            steps=_append_step(
                event.steps, "project-synthesis", self.clock, started_at
            ),
        )

    @step
    async def render_documents(
        self, event: ProjectsGeneratedEvent
    ) -> DocumentsRenderedEvent:
        """Render validated artifacts into deterministic Markdown.

        Args:
            event:
                All validated structured artifacts.

        Returns:
            Rendered documents and their source artifacts.
        """

        started_at = self.clock()
        with application_span(
            "portfolio.rendering",
            "CHAIN",
            {"artifact.count": len(event.components) + len(event.projects)},
        ):
            documents = render_balanced_documents(
                event.project.slug,
                event.project.display_name,
                event.components,
                event.projects,
            )
        return DocumentsRenderedEvent(
            project=event.project,
            generation=event.generation,
            snapshot=event.snapshot,
            artifacts=(*event.components, *event.projects),
            documents=documents,
            run_started_at=event.run_started_at,
            steps=_append_step(event.steps, "rendering", self.clock, started_at),
        )

    @step
    async def build_manifest(
        self, event: DocumentsRenderedEvent
    ) -> CorpusValidatedEvent:
        """Construct the stable manifest for the complete in-memory corpus.

        Publication performs the full filesystem corpus validation before making
        the release visible.

        Args:
            event:
                Complete deterministic rendered document set.

        Returns:
            Manifest-backed in-memory corpus ready for atomic publication.
        """

        started_at = self.clock()
        with application_span(
            "portfolio.manifest", "CHAIN", {"document.count": len(event.documents)}
        ):
            manifest = build_corpus_manifest(
                project_display_name=event.project.display_name,
                snapshot=event.snapshot,
                generation=event.generation,
                documents=event.documents,
            )
            validate_rendered_corpus(manifest, event.documents, event.snapshot)
        return CorpusValidatedEvent(
            project=event.project,
            generation=event.generation,
            snapshot=event.snapshot,
            artifacts=event.artifacts,
            documents=event.documents,
            manifest=manifest,
            run_started_at=event.run_started_at,
            steps=_append_step(
                event.steps, "manifest-validation", self.clock, started_at
            ),
        )

    @step
    async def publish_and_report(self, event: CorpusValidatedEvent) -> StopEvent:
        """Atomically publish the corpus and assemble current-run metrics.

        Args:
            event:
                Complete manifest-backed corpus.

        Returns:
            Terminal event carrying the typed successful run report.
        """

        started_at = self.clock()
        with application_span(
            "portfolio.publication",
            "CHAIN",
            {"corpus.fingerprint": event.manifest.corpus_fingerprint},
        ):
            publication = self.publisher(
                publication_path=event.generation.locations.publication,
                manifest=event.manifest,
                documents=event.documents,
                snapshot=event.snapshot,
            )
        steps = _append_step(event.steps, "publication", self.clock, started_at)
        artifact_reports = tuple(artifact.run_report for artifact in event.artifacts)
        provider_reports = tuple(
            report for report in artifact_reports if not report.cache_hit
        )
        report = GenerationRunReport(
            project_slug=event.project.slug,
            resolved_commit_sha=event.snapshot.resolved_commit_sha,
            source_fingerprint=event.snapshot.source_fingerprint,
            generation_configuration_fingerprint=(
                generation_configuration_fingerprint(event.generation)
            ),
            corpus_fingerprint=publication.corpus_fingerprint,
            artifacts=artifact_reports,
            provider_call_count=len(provider_reports),
            billed_token_usage=_aggregate_usage(provider_reports),
            estimated_cost=_aggregate_cost(provider_reports),
            elapsed_seconds=max(0.0, self.clock() - event.run_started_at),
            steps=steps,
            published_path=publication.publication_path,
            release_path=publication.release_path,
        )
        return StopEvent(result=report)


async def run_portfolio_corpus_workflow(
    workflow: PortfolioCorpusWorkflow,
    *,
    config_path: Path,
    project_slug: str,
) -> GenerationRunReport:
    """Run one configured Portfolio corpus workflow to completion.

    Args:
        workflow:
            Fully composed workflow with injected boundaries.
        config_path:
            Portfolio YAML configuration path.
        project_slug:
            Exact configured project identifier.

    Returns:
        Typed final report for a successfully published corpus.
    """

    result = await workflow.run(
        start_event=GenerateCorpusStartEvent(
            config_path=config_path,
            project_slug=project_slug,
        )
    )
    if not isinstance(result, GenerationRunReport):
        raise TypeError("portfolio workflow returned an unexpected result")

    return result


def _step_report(
    name: str,
    clock: Callable[[], float],
    started_at: float,
) -> StepRunReport:
    """Build a non-negative workflow-step latency report.

    Args:
        name:
            Stable step name.
        clock:
            Monotonic clock.
        started_at:
            Clock value captured before the step.

    Returns:
        Immutable step report.
    """

    return StepRunReport(
        name=name,
        latency_seconds=max(0.0, clock() - started_at),
    )


def _append_step(
    steps: tuple[StepRunReport, ...],
    name: str,
    clock: Callable[[], float],
    started_at: float,
) -> tuple[StepRunReport, ...]:
    """Append one timed step to immutable accumulated run state.

    Args:
        steps:
            Previously completed step reports.
        name:
            Stable new step name.
        clock:
            Monotonic clock.
        started_at:
            Clock value captured before the step.

    Returns:
        New ordered immutable step tuple.
    """

    return (*steps, _step_report(name, clock, started_at))


def _aggregate_usage(reports: Sequence[ArtifactRunReport]) -> TokenUsage:
    """Aggregate billed usage without converting unavailable counts to zero.

    Args:
        reports:
            Current-run reports for actual provider calls.

    Returns:
        Aggregate token usage, preserving unavailable dimensions as null.
    """

    if not reports:
        return TokenUsage(input_tokens=0, output_tokens=0)
    input_values = tuple(report.token_usage.input_tokens for report in reports)
    output_values = tuple(report.token_usage.output_tokens for report in reports)
    return TokenUsage(
        input_tokens=(
            sum(value for value in input_values if value is not None)
            if all(value is not None for value in input_values)
            else None
        ),
        output_tokens=(
            sum(value for value in output_values if value is not None)
            if all(value is not None for value in output_values)
            else None
        ),
    )


def _aggregate_cost(reports: Sequence[ArtifactRunReport]) -> Decimal | None:
    """Aggregate current-run cost only when every provider call is priced.

    Args:
        reports:
            Current-run reports for actual provider calls.

    Returns:
        Total estimated cost, zero for no calls, or null when incomplete.
    """

    if not reports:
        return Decimal(0)
    costs = tuple(report.estimated_cost for report in reports)
    if any(cost is None for cost in costs):
        return None
    return sum((cost for cost in costs if cost is not None), start=Decimal(0))
