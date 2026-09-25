"""Port for validated structured-artifact persistence."""

from __future__ import annotations

from typing import Protocol

from genai_template.workflow.portfolio.domain.generation import (
    ArtifactKind,
    CachedArtifact,
)
from genai_template.workflow.portfolio.domain.summaries import StructuredSummary


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
