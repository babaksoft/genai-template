"""Typed events for the Portfolio corpus LlamaIndex workflow."""

from __future__ import annotations

from pathlib import Path

from llama_index.core.workflow import Event, StartEvent

from genai_template.workflow.portfolio.config.models import (
    GenerationConfig,
    ProjectConfig,
)
from genai_template.workflow.portfolio.domain.components import ComponentSummaryArtifact
from genai_template.workflow.portfolio.domain.generation import RenderedDocument
from genai_template.workflow.portfolio.domain.manifest import CorpusManifest
from genai_template.workflow.portfolio.domain.projects import ProjectSummaryArtifact
from genai_template.workflow.portfolio.domain.reports import StepRunReport
from genai_template.workflow.portfolio.domain.snapshot import RepositorySnapshot
from genai_template.workflow.portfolio.domain.summary import SummaryPlan


class GenerateCorpusStartEvent(StartEvent):
    """Request to generate one configured project's corpus.

    Attributes:
        config_path:
            Portfolio configuration path.
        project_slug:
            Exact configured project identifier.
    """

    config_path: Path
    project_slug: str


class ConfigurationSelectedEvent(Event):
    """Validated generation configuration and selected project.

    Attributes:
        project:
            Exactly selected project configuration.
        generation:
            Required balanced generation configuration.
        run_started_at:
            Monotonic start value for the complete run.
        steps:
            Completed workflow-step reports.
    """

    project: ProjectConfig
    generation: GenerationConfig
    run_started_at: float
    steps: tuple[StepRunReport, ...]


class SnapshotSelectedEvent(Event):
    """Canonical committed snapshot selected for generation.

    Attributes:
        project:
            Selected project configuration.
        generation:
            Balanced generation configuration.
        snapshot:
            Canonical selected committed snapshot.
        run_started_at:
            Monotonic start value for the complete run.
        steps:
            Completed workflow-step reports.
    """

    project: ProjectConfig
    generation: GenerationConfig
    snapshot: RepositorySnapshot
    run_started_at: float
    steps: tuple[StepRunReport, ...]


class SummaryPlannedEvent(Event):
    """Deterministic logical summary plan ready for execution.

    Attributes:
        project:
            Selected project configuration.
        generation:
            Balanced generation configuration.
        snapshot:
            Canonical selected committed snapshot.
        plan:
            Deterministic logical summary plan.
        run_started_at:
            Monotonic start value for the complete run.
        steps:
            Completed workflow-step reports.
    """

    project: ProjectConfig
    generation: GenerationConfig
    snapshot: RepositorySnapshot
    plan: SummaryPlan
    run_started_at: float
    steps: tuple[StepRunReport, ...]


class ComponentsGeneratedEvent(Event):
    """Validated component artifacts generated or reused.

    Attributes:
        project:
            Selected project configuration.
        generation:
            Balanced generation configuration.
        snapshot:
            Canonical selected committed snapshot.
        components:
            Ordered validated component artifacts.
        run_started_at:
            Monotonic start value for the complete run.
        steps:
            Completed workflow-step reports.
    """

    project: ProjectConfig
    generation: GenerationConfig
    snapshot: RepositorySnapshot
    components: tuple[ComponentSummaryArtifact, ...]
    run_started_at: float
    steps: tuple[StepRunReport, ...]


class ProjectsGeneratedEvent(Event):
    """Validated project synthesis artifacts generated or reused.

    Attributes:
        project:
            Selected project configuration.
        generation:
            Balanced generation configuration.
        snapshot:
            Canonical selected committed snapshot.
        components:
            Ordered validated component artifacts.
        projects:
            Ordered validated project artifacts.
        run_started_at:
            Monotonic start value for the complete run.
        steps:
            Completed workflow-step reports.
    """

    project: ProjectConfig
    generation: GenerationConfig
    snapshot: RepositorySnapshot
    components: tuple[ComponentSummaryArtifact, ...]
    projects: tuple[ProjectSummaryArtifact, ...]
    run_started_at: float
    steps: tuple[StepRunReport, ...]


class DocumentsRenderedEvent(Event):
    """Deterministic Markdown documents ready for manifest construction.

    Attributes:
        project:
            Selected project configuration.
        generation:
            Balanced generation configuration.
        snapshot:
            Canonical selected committed snapshot.
        artifacts:
            Ordered artifacts supplying run metrics and provenance.
        documents:
            Deterministically rendered Markdown documents.
        run_started_at:
            Monotonic start value for the complete run.
        steps:
            Completed workflow-step reports.
    """

    project: ProjectConfig
    generation: GenerationConfig
    snapshot: RepositorySnapshot
    artifacts: tuple[ComponentSummaryArtifact | ProjectSummaryArtifact, ...]
    documents: tuple[RenderedDocument, ...]
    run_started_at: float
    steps: tuple[StepRunReport, ...]


class CorpusValidatedEvent(Event):
    """Complete in-memory corpus validated and ready for publication.

    Attributes:
        project:
            Selected project configuration.
        generation:
            Balanced generation configuration.
        snapshot:
            Canonical selected committed snapshot.
        artifacts:
            Ordered artifacts supplying run metrics and provenance.
        documents:
            Complete deterministic Markdown document set.
        manifest:
            Stable manifest for the complete corpus.
        run_started_at:
            Monotonic start value for the complete run.
        steps:
            Completed workflow-step reports.
    """

    project: ProjectConfig
    generation: GenerationConfig
    snapshot: RepositorySnapshot
    artifacts: tuple[ComponentSummaryArtifact | ProjectSummaryArtifact, ...]
    documents: tuple[RenderedDocument, ...]
    manifest: CorpusManifest
    run_started_at: float
    steps: tuple[StepRunReport, ...]
