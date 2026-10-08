"""Provider-neutral failures for Portfolio workflow boundaries."""

from pathlib import Path


class RepositoryReadError(RuntimeError):
    """Failure to read an immutable repository revision.

    Attributes:
        repository_path:
            Configured local repository path involved in the failure.
        requested_ref:
            Configured ref involved in the failure, when ref resolution had begun.
        operation:
            Short description of the failed read operation.
    """

    def __init__(
        self,
        message: str,
        *,
        repository_path: Path,
        requested_ref: str | None = None,
        operation: str,
    ) -> None:
        """Initialize a contextual repository read failure.

        Args:
            message:
                Human-readable description of the failure.
            repository_path:
                Configured path involved in the failure.
            requested_ref:
                Ref involved in the failure, when applicable.
            operation:
                Short description of the failed read operation.
        """

        super().__init__(message)
        self.repository_path = repository_path
        self.requested_ref = requested_ref
        self.operation = operation


class TextGenerationError(RuntimeError):
    """Provider-neutral text generation boundary failure.

    Attributes:
        provider:
            Stable provider identifier.
        model:
            Provider model involved in the failure.
        reason:
            Stable short reason suitable for workflow handling.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        model: str,
        reason: str,
    ) -> None:
        """Initialize a text generation failure.

        Args:
            message:
                Human-readable failure description without sensitive content.
            provider:
                Provider involved in the failure.
            model:
                Model involved in the failure.
            reason:
                Stable failure category.
        """

        super().__init__(message)
        self.provider = provider
        self.model = model
        self.reason = reason


class ArtifactCacheError(RuntimeError):
    """Failure to read, validate, or atomically write a cached artifact.

    Attributes:
        generation_fingerprint:
            Stable cache key involved in the failure.
        reason:
            Stable short reason suitable for workflow handling.
    """

    def __init__(
        self,
        message: str,
        *,
        generation_fingerprint: str,
        reason: str,
    ) -> None:
        """Initialize a cache failure without exposing machine-local paths.

        Args:
            message:
                Human-readable failure description without cached content.
            generation_fingerprint:
                Stable cache key involved in the failure.
            reason:
                Stable failure category.
        """

        super().__init__(message)
        self.generation_fingerprint = generation_fingerprint
        self.reason = reason


class ArtifactValidationError(RuntimeError):
    """Failure to reconcile generated output with requested provenance.

    Attributes:
        generation_fingerprint:
            Stable generation identity involved in the failure.
        reason:
            Stable short reason suitable for workflow handling.
    """

    def __init__(
        self,
        message: str,
        *,
        generation_fingerprint: str,
        reason: str,
    ) -> None:
        """Initialize a generated-artifact validation failure.

        Args:
            message:
                Human-readable failure description without provider content.
            generation_fingerprint:
                Stable generation identity involved in the failure.
            reason:
                Stable failure category.
        """

        super().__init__(message)
        self.generation_fingerprint = generation_fingerprint
        self.reason = reason


class EvidenceValidationError(ValueError):
    """A generated summary cites evidence outside its exact input scope.

    Attributes:
        invalid_paths:
            Sorted paths that were not members of the permitted scope.
        permitted_paths:
            Sorted exact paths permitted for this generation call.
    """

    def __init__(
        self,
        message: str,
        *,
        invalid_paths: tuple[str, ...],
        permitted_paths: tuple[str, ...],
    ) -> None:
        """Initialize an evidence validation failure.

        Args:
            message:
                Human-readable failure description.
            invalid_paths:
                Paths cited outside the allowed scope.
            permitted_paths:
                Exact allowed evidence paths.
        """

        super().__init__(message)
        self.invalid_paths = invalid_paths
        self.permitted_paths = permitted_paths


class SnapshotSelectionError(ValueError):
    """Failure to select or normalize a committed repository entry.

    Attributes:
        path:
            Repository-relative path involved in the failure, when available.
        reason:
            Stable short reason identifying the rejected condition.
    """

    def __init__(self, message: str, *, path: str | None, reason: str) -> None:
        """Initialize a contextual snapshot-selection failure.

        Args:
            message:
                Human-readable description of the failure.
            path:
                Repository-relative path involved in the failure, when available.
            reason:
                Stable short reason identifying the rejected condition.
        """

        super().__init__(message)
        self.path = path
        self.reason = reason


class SummaryPlanningError(ValueError):
    """Failure to derive configured logical units from a snapshot.

    Attributes:
        unit_id:
            Logical unit involved in the failure, when one is available.
        reason:
            Stable short reason identifying the invalid plan condition.
    """

    def __init__(self, message: str, *, unit_id: str | None, reason: str) -> None:
        """Initialize a contextual summary-planning failure.

        Args:
            message:
                Human-readable description of the failure.
            unit_id:
                Logical unit involved in the failure, when available.
            reason:
                Stable short reason identifying the invalid condition.
        """

        super().__init__(message)
        self.unit_id = unit_id
        self.reason = reason
