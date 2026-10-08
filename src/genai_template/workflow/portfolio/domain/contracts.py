from __future__ import annotations

from typing import Protocol, runtime_checkable

from genai_template.workflow.portfolio.config.models import LocalGitRepositoryConfig
from genai_template.workflow.portfolio.domain.artifacts import (
    ArtifactKind,
    CachedArtifact,
)
from genai_template.workflow.portfolio.domain.generation import (
    GenerationRequest,
    TextGenerationResponse,
)
from genai_template.workflow.portfolio.domain.snapshot import RepositoryReadResult
from genai_template.workflow.portfolio.domain.summaries import StructuredSummary


@runtime_checkable
class RepositoryReader(Protocol):
    """Provider-neutral interface for reading one immutable repository revision."""

    def read(self, repository: LocalGitRepositoryConfig) -> RepositoryReadResult:
        """Read committed entries and identity for a configured repository.

        Args:
            repository:
                Validated repository source and ref settings.

        Returns:
            Immutable commit identity and committed repository entries.

        Raises:
            RepositoryReadError:
                If the repository or configured revision cannot be read.
        """

        ...


class ArtifactCache(Protocol):
    """Persistence boundary for validated generation artifacts."""

    def get[SummaryT: StructuredSummary](
        self,
        generation_fingerprint: str,
        *,
        expected_kind: ArtifactKind,
        output_schema_version: str,
        output_type: type[SummaryT],
    ) -> CachedArtifact | None:
        """Load and fully validate one cache entry when it exists.

        Args:
            generation_fingerprint:
                Full generation fingerprint used as the cache key.
            expected_kind:
                Artifact kind required by the caller.
            output_schema_version:
                Structured-output schema version required by the caller.
            output_type:
                Pydantic summary model used to validate stored output.

        Returns:
            A validated artifact, or null when the key has never been stored.

        Raises:
            ArtifactCacheError:
                If an existing entry cannot be read or validated.
        """

        ...

    def put[SummaryT: StructuredSummary](
        self,
        artifact: CachedArtifact,
        *,
        output_type: type[SummaryT],
    ) -> None:
        """Validate and atomically persist one cache entry.

        Args:
            artifact:
                Complete validated cache envelope to persist.
            output_type:
                Pydantic summary model used to validate stored output.

        Raises:
            ArtifactCacheError:
                If validation or the atomic write fails.
        """

        ...


class TextGenerator(Protocol):
    """Provider-neutral interface for one-call plain-text generation."""

    def generate(self, request: GenerationRequest) -> TextGenerationResponse:
        """Generate one complete plain-text response.

        Args:
            request:
                Fully identified prompt request.

        Returns:
            Plain text plus provider identity and accounting metadata.

        Raises:
            TextGenerationError:
                If request identity, transport, or completion validation fails.
        """

        ...
