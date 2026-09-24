"""Canonical domain models for portfolio repository snapshots."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _ImmutableDomainModel(BaseModel):
    """Base model for immutable values crossing workflow boundaries."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class CommittedRepositoryEntry(_ImmutableDomainModel):
    """One entry read from an immutable repository commit.

    Blob entries carry their bytes. Gitlink entries identify submodules and do not
    carry content; selection can consequently reject them without traversing into
    another repository.

    Attributes:
        path:
            Normalized repository-relative POSIX path of the committed entry.
        mode:
            Six-digit Git file mode reported for the committed entry.
        object_type:
            Git object type distinguishing blobs from submodule gitlinks.
        object_id:
            Full hexadecimal Git object identifier.
        content:
            Committed blob bytes, or null for a submodule gitlink.
    """

    path: str = Field(
        min_length=1,
        description="Normalized repository-relative POSIX path.",
    )
    mode: str = Field(
        pattern=r"^[0-7]{6}$",
        description="Six-digit Git file mode.",
    )
    object_type: Literal["blob", "commit"] = Field(
        description="Git object type for a blob or submodule gitlink."
    )
    object_id: str = Field(
        pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$",
        description="Full hexadecimal Git object identifier.",
    )
    content: bytes | None = Field(
        default=None,
        description="Committed blob bytes, or null for a submodule gitlink.",
    )

    @model_validator(mode="after")
    def validate_content(self) -> CommittedRepositoryEntry:
        """Ensure content is present only for blob entries.

        Returns:
            The validated committed entry.

        Raises:
            ValueError:
                If blob content is absent or a gitlink carries content.
        """

        if self.object_type == "blob" and self.content is None:
            raise ValueError("content is required for blob entries")
        if self.object_type == "commit" and self.content is not None:
            raise ValueError("content is not supported for gitlink entries")
        return self


class SnapshotFile(_ImmutableDomainModel):
    """A normalized UTF-8 text file selected from a repository commit.

    Attributes:
        path:
            Normalized repository-relative POSIX path.
        text:
            UTF-8 text normalized according to the snapshot contract.
        content_hash:
            SHA-256 hash of the normalized UTF-8 content.
        byte_size:
            Size of the normalized UTF-8 content in bytes.
    """

    path: str = Field(
        min_length=1,
        description="Normalized repository-relative POSIX path.",
    )
    text: str = Field(description="Normalized UTF-8 file content.")
    content_hash: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 hash of the normalized UTF-8 content.",
    )
    byte_size: int = Field(
        ge=0,
        description="Size of the normalized UTF-8 content in bytes.",
    )


class RepositorySnapshot(_ImmutableDomainModel):
    """Canonical selected contents and identity of one repository commit.

    Attributes:
        project_slug:
            Stable project identifier from the portfolio configuration.
        repository_url:
            Normalized repository URL when one is available.
        requested_ref:
            Repository ref requested before immutable commit resolution.
        resolved_commit_sha:
            Full object identifier of the resolved commit.
        source_fingerprint:
            SHA-256 identity of the stable selection settings and contents.
        files:
            Selected normalized files in deterministic path order.
    """

    project_slug: str = Field(
        min_length=1,
        description="Stable project identifier from portfolio configuration.",
    )
    repository_url: str | None = Field(
        default=None,
        description="Normalized repository URL when available.",
    )
    requested_ref: str = Field(
        min_length=1,
        description="Repository ref requested before commit resolution.",
    )
    resolved_commit_sha: str = Field(
        pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$",
        description="Full hexadecimal identifier of the resolved commit.",
    )
    source_fingerprint: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 identity of the selected normalized source.",
    )
    files: tuple[SnapshotFile, ...] = Field(
        description="Selected normalized files in deterministic path order."
    )


class SummaryUnitPlan(_ImmutableDomainModel):
    """The exact snapshot files used to produce one logical summary.

    Attributes:
        unit_id:
            Stable identifier of the configured logical summary unit.
        input_fingerprint:
            SHA-256 identity of this unit's settings and selected contents.
        files:
            Non-empty ordered snapshot files assigned to the unit.
    """

    unit_id: str = Field(
        min_length=1,
        description="Stable logical summary-unit identifier.",
    )
    input_fingerprint: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 identity of the unit settings and contents.",
    )
    files: tuple[SnapshotFile, ...] = Field(
        min_length=1,
        description="Ordered snapshot files assigned to this summary unit.",
    )


class SummaryPlan(_ImmutableDomainModel):
    """Stable logical summary work derived from a repository snapshot.

    Attributes:
        project_slug:
            Stable project identifier shared with the source snapshot.
        resolved_commit_sha:
            Full object identifier of the snapshot's resolved commit.
        source_fingerprint:
            SHA-256 identity shared with the source snapshot.
        units:
            Non-empty ordered logical summary units to process.
    """

    project_slug: str = Field(
        min_length=1,
        description="Stable project identifier shared with the snapshot.",
    )
    resolved_commit_sha: str = Field(
        pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$",
        description="Full hexadecimal identifier of the resolved commit.",
    )
    source_fingerprint: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 identity shared with the source snapshot.",
    )
    units: tuple[SummaryUnitPlan, ...] = Field(
        min_length=1,
        description="Ordered logical summary units to process.",
    )
