"""Tests for deterministic logical summary planning."""

from __future__ import annotations

import hashlib

import pytest

from genai_template.workflow.portfolio import (
    RepositorySnapshot,
    SnapshotFile,
    SummaryPlanningError,
    SummaryUnitConfig,
    build_summary_plan,
)


def _file(path: str, text: str) -> SnapshotFile:
    """Build a canonical snapshot file for planning tests.

    Args:
        path:
            Repository-relative path.
        text:
            Canonical file text.

    Returns:
        Snapshot file with a matching content hash and byte size.
    """

    content = text.encode("utf-8")
    return SnapshotFile(
        path=path,
        text=text,
        content_hash=hashlib.sha256(content).hexdigest(),
        byte_size=len(content),
    )


def _snapshot(*files: SnapshotFile) -> RepositorySnapshot:
    """Build a canonical repository snapshot for planning tests.

    Args:
        *files:
            Files to expose in canonical path order.

    Returns:
        Repository snapshot carrying the supplied files.
    """

    return RepositorySnapshot(
        project_slug="sample-project",
        repository_url=None,
        requested_ref="HEAD",
        resolved_commit_sha="a" * 40,
        source_fingerprint="b" * 64,
        files=files,
    )


def _unit(unit_id: str, *paths: str) -> SummaryUnitConfig:
    """Build validated logical-unit settings.

    Args:
        unit_id:
            Stable logical unit identifier.
        *paths:
            Membership patterns.

    Returns:
        Validated summary-unit configuration.
    """

    return SummaryUnitConfig(id=unit_id, paths=paths)


def test_unit_membership_order_and_overlap_are_deterministic() -> None:
    """Units retain configuration order and may deliberately share files."""

    snapshot = _snapshot(
        _file("README.md", "read me\n"),
        _file("src/a.py", "a\n"),
        _file("src/pkg/b.py", "b\n"),
        _file("tests/test_a.py", "test\n"),
    )

    plan = build_summary_plan(
        snapshot,
        (
            _unit("python", "src/**/*.py", "tests/**/*.py"),
            _unit("application", "README.md", "src/**/*.py"),
        ),
    )

    assert [unit.unit_id for unit in plan.units] == ["python", "application"]
    assert [file.path for file in plan.units[0].files] == [
        "src/a.py",
        "src/pkg/b.py",
        "tests/test_a.py",
    ]
    assert [file.path for file in plan.units[1].files] == [
        "README.md",
        "src/a.py",
        "src/pkg/b.py",
    ]
    assert plan.units[0].files[0] is plan.units[1].files[1]


def test_empty_unit_is_rejected_with_context() -> None:
    """A configured unit must resolve to at least one selected snapshot file."""

    with pytest.raises(SummaryPlanningError) as caught:
        build_summary_plan(
            _snapshot(_file("README.md", "read me\n")),
            (_unit("python", "src/**/*.py"),),
        )

    assert caught.value.unit_id == "python"
    assert caught.value.reason == "empty-unit"


def test_duplicate_unit_ids_are_defensively_rejected() -> None:
    """Planning rejects duplicates even if a caller bypasses model validation."""

    duplicate = _unit("python", "src/**/*.py")

    with pytest.raises(SummaryPlanningError) as caught:
        build_summary_plan(
            _snapshot(_file("src/a.py", "a\n")),
            (duplicate, duplicate),
        )

    assert caught.value.reason == "duplicate-unit-id"


def test_unit_fingerprint_is_stable_across_equivalent_pattern_order() -> None:
    """Semantically equivalent membership pattern order has one identity."""

    snapshot = _snapshot(
        _file("src/a.py", "a\n"),
        _file("tests/test_a.py", "test\n"),
    )
    first = build_summary_plan(
        snapshot,
        (_unit("python", "src/**/*.py", "tests/**/*.py"),),
    )
    second = build_summary_plan(
        snapshot,
        (_unit("python", "tests/**/*.py", "src/**/*.py"),),
    )

    assert first.units[0].input_fingerprint == second.units[0].input_fingerprint


def test_unit_fingerprints_are_isolated_from_unrelated_files() -> None:
    """Changing another unit's file does not invalidate an unaffected unit."""

    first_snapshot = _snapshot(
        _file("src/a.py", "stable\n"),
        _file("tests/test_a.py", "first\n"),
    )
    second_snapshot = _snapshot(
        _file("src/a.py", "stable\n"),
        _file("tests/test_a.py", "second\n"),
    )
    units = (
        _unit("application", "src/**/*.py"),
        _unit("tests", "tests/**/*.py"),
    )

    first = build_summary_plan(first_snapshot, units)
    second = build_summary_plan(second_snapshot, units)

    assert first.units[0].input_fingerprint == second.units[0].input_fingerprint
    assert first.units[1].input_fingerprint != second.units[1].input_fingerprint


def test_unit_identifier_and_patterns_affect_fingerprint() -> None:
    """Stable unit configuration is part of the Stage 0 input identity."""

    snapshot = _snapshot(_file("src/a.py", "a\n"))
    baseline = build_summary_plan(snapshot, (_unit("application", "src/**/*.py"),))
    changed_id = build_summary_plan(snapshot, (_unit("core", "src/**/*.py"),))
    changed_pattern = build_summary_plan(snapshot, (_unit("application", "**/*.py"),))

    assert baseline.units[0].input_fingerprint != changed_id.units[0].input_fingerprint
    assert (
        baseline.units[0].input_fingerprint
        != changed_pattern.units[0].input_fingerprint
    )
