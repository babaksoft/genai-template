"""Repository-reader adapters for Portfolio workflows."""

from genai_template.workflow.portfolio.adapters.repositories.local_git import (
    LocalGitSnapshotReader,
)

__all__ = ["LocalGitSnapshotReader"]
