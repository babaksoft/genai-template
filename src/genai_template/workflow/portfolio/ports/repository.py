"""Provider-neutral repository-reading port for Portfolio workflows."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from genai_template.workflow.portfolio.config.models import LocalGitRepositoryConfig
from genai_template.workflow.portfolio.domain.snapshot import RepositoryReadResult


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
