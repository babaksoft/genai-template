"""Index-build lifecycle contracts."""

from enum import StrEnum


class IndexBuildStatus(StrEnum):
    """Allowed durable states for an index-build attempt."""

    BUILDING = "building"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
