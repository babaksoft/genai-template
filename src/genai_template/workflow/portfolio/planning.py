"""Deterministic logical summary planning for repository snapshots."""

from __future__ import annotations

import hashlib
import json

from genai_template.workflow.portfolio.config import SummaryUnitConfig
from genai_template.workflow.portfolio.models import (
    RepositorySnapshot,
    SnapshotFile,
    SummaryPlan,
    SummaryUnitPlan,
)
from genai_template.workflow.portfolio.selection import matches_repository_pattern


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


def build_summary_plan(
    snapshot: RepositorySnapshot,
    summary_units: tuple[SummaryUnitConfig, ...],
) -> SummaryPlan:
    """Resolve configured logical units against a canonical snapshot.

    Unit order follows configuration order and member files retain snapshot path
    order. A file may deliberately belong to more than one unit; membership does
    not remove it from later units.

    Args:
        snapshot:
            Canonical selected repository snapshot.
        summary_units:
            Explicit configured logical units.

    Returns:
        Stable summary plan containing non-empty logical units.

    Raises:
        SummaryPlanningError:
            If units are absent, identifiers are duplicated, or a unit matches no
            selected snapshot files.
    """

    if not summary_units:
        raise SummaryPlanningError(
            "At least one summary unit is required",
            unit_id=None,
            reason="no-units",
        )

    unit_ids = [unit.id for unit in summary_units]
    if len(unit_ids) != len(set(unit_ids)):
        raise SummaryPlanningError(
            "Summary unit identifiers must be unique",
            unit_id=None,
            reason="duplicate-unit-id",
        )

    planned_units: list[SummaryUnitPlan] = []
    for unit in summary_units:
        files = tuple(
            file
            for file in snapshot.files
            if any(
                matches_repository_pattern(file.path, pattern) for pattern in unit.paths
            )
        )
        if not files:
            raise SummaryPlanningError(
                f"Summary unit {unit.id!r} matches no selected files",
                unit_id=unit.id,
                reason="empty-unit",
            )
        planned_units.append(
            SummaryUnitPlan(
                unit_id=unit.id,
                input_fingerprint=_unit_fingerprint(unit, files),
                files=files,
            )
        )

    return SummaryPlan(
        project_slug=snapshot.project_slug,
        resolved_commit_sha=snapshot.resolved_commit_sha,
        source_fingerprint=snapshot.source_fingerprint,
        units=tuple(planned_units),
    )


def _unit_fingerprint(unit: SummaryUnitConfig, files: tuple[SnapshotFile, ...]) -> str:
    """Calculate one logical unit's canonical input identity.

    Args:
        unit:
            Logical unit identifier and membership patterns.
        files:
            Matching snapshot files in canonical path order.

    Returns:
        Hexadecimal SHA-256 input fingerprint.
    """

    payload = {
        "files": [
            {"content_hash": file.content_hash, "path": file.path} for file in files
        ],
        "paths": sorted(set(unit.paths)),
        "unit_id": unit.id,
        "version": 1,
    }
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
