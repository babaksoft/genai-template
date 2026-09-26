"""Portfolio workflow domain models and failures."""

from genai_template.workflow.portfolio.domain.components import ComponentSummaryArtifact
from genai_template.workflow.portfolio.domain.errors import (
    ArtifactCacheError,
    ArtifactValidationError,
    RepositoryReadError,
    StructuredGenerationError,
)
from genai_template.workflow.portfolio.domain.generation import (
    ArtifactKind,
    ArtifactProvenance,
    CachedArtifact,
    CorpusBuild,
    GenerationRequest,
    GenerationResult,
    ProviderAuditMetadata,
    RenderedDocument,
    TokenUsage,
)
from genai_template.workflow.portfolio.domain.manifest import (
    CorpusManifest,
    ManifestDocument,
    ManifestPrompt,
)
from genai_template.workflow.portfolio.domain.projects import (
    ProjectArtifactKind,
    ProjectSummaryArtifact,
)
from genai_template.workflow.portfolio.domain.reports import (
    ArtifactRunReport,
    GenerationRunReport,
)
from genai_template.workflow.portfolio.domain.snapshot import (
    CommittedRepositoryEntry,
    RepositoryReadResult,
    RepositorySnapshot,
    SnapshotFile,
)
from genai_template.workflow.portfolio.domain.summaries import (
    OUTPUT_SCHEMA_VERSION,
    ArchitectureSummary,
    ComponentSummary,
    EvidenceSection,
    ProjectOverviewSummary,
    StructuredSummary,
    TestingOperationsSummary,
    summary_evidence_paths,
)
from genai_template.workflow.portfolio.domain.summary import (
    SummaryPlan,
    SummaryUnitPlan,
)

__all__ = [
    "OUTPUT_SCHEMA_VERSION",
    "ArchitectureSummary",
    "ArtifactCacheError",
    "ArtifactKind",
    "ArtifactProvenance",
    "ArtifactRunReport",
    "ArtifactValidationError",
    "CachedArtifact",
    "CommittedRepositoryEntry",
    "ComponentSummary",
    "ComponentSummaryArtifact",
    "CorpusBuild",
    "CorpusManifest",
    "EvidenceSection",
    "GenerationRequest",
    "GenerationResult",
    "GenerationRunReport",
    "ManifestDocument",
    "ManifestPrompt",
    "ProjectArtifactKind",
    "ProjectOverviewSummary",
    "ProjectSummaryArtifact",
    "ProviderAuditMetadata",
    "RenderedDocument",
    "RepositoryReadError",
    "RepositoryReadResult",
    "RepositorySnapshot",
    "SnapshotFile",
    "StructuredGenerationError",
    "StructuredSummary",
    "SummaryPlan",
    "SummaryUnitPlan",
    "TestingOperationsSummary",
    "TokenUsage",
    "summary_evidence_paths",
]
