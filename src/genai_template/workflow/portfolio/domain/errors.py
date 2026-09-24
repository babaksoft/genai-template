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
