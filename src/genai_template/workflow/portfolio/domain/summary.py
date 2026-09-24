"""Canonical domain models for Portfolio summary planning."""

from __future__ import annotations

from pydantic import Field

from genai_template.workflow.portfolio.domain.snapshot import (
    SnapshotFile,
    _ImmutableDomainModel,
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
