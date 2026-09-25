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


class StructuredGenerationError(RuntimeError):
    """Provider-neutral structured generation boundary failure.

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
        """Initialize a structured generation failure.

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
