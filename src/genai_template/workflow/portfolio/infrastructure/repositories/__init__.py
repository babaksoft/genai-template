"""Repository-reader adapters for Portfolio workflows."""

from genai_template.workflow.portfolio.infrastructure.repositories.local_git import (
    LocalGitSnapshotReader,
    matches_repository_pattern,
)

__all__ = [
    "LocalGitSnapshotReader",
    "matches_repository_pattern",
]
