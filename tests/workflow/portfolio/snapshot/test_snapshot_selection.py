"""Tests for deterministic repository snapshot-selection behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from genai_template.workflow.portfolio import (
    CommittedRepositoryEntry,
    RepositoryReadResult,
    SelectionConfig,
    SnapshotSelectionError,
    build_repository_snapshot,
)
from genai_template.workflow.portfolio.snapshot.selection import (
    matches_repository_pattern,
)

_COMMIT_SHA = "a" * 40
_OBJECT_ID = "b" * 40


def _entry(
    path: str,
    content: bytes | None = b"text\n",
    *,
    mode: str = "100644",
    object_type: str = "blob",
) -> CommittedRepositoryEntry:
    """Build one committed entry for selection tests.

    Args:
        path:
            Repository-relative entry path.
        content:
            Blob content, or null for a gitlink.
        mode:
            Six-digit Git file mode.
        object_type:
            Git object type accepted by the domain model.

    Returns:
        Validated committed repository entry.
    """

    return CommittedRepositoryEntry(
        path=path,
        mode=mode,
        object_type=object_type,  # type: ignore[arg-type]
        object_id=_OBJECT_ID,
        content=content,
    )


def _read_result(
    *entries: CommittedRepositoryEntry,
    repository_url: str | None = None,
) -> RepositoryReadResult:
    """Build a provider-neutral read result for selection tests.

    Args:
        *entries:
            Committed entries to expose to selection.
        repository_url:
            Optional provider repository URL.

    Returns:
        Validated repository read result.
    """

    return RepositoryReadResult(
        repository_url=repository_url,
        requested_ref="HEAD",
        resolved_commit_sha=_COMMIT_SHA,
        entries=entries,
    )


def _selection(
    *,
    include: tuple[str, ...] = ("**/*.py",),
    exclude: tuple[str, ...] = (),
    max_file_bytes: int = 1024,
) -> SelectionConfig:
    """Build validated file-selection settings.

    Args:
        include:
            Candidate file patterns.
        exclude:
            Patterns removed after inclusion.
        max_file_bytes:
            Maximum accepted original blob size.

    Returns:
        Validated selection configuration.
    """

    return SelectionConfig(
        include=include,
        exclude=exclude,
        max_file_bytes=max_file_bytes,
    )


@pytest.mark.parametrize(
    ("path", "pattern", "expected"),
    [
        ("README.md", "README.md", True),
        ("docs/README.md", "README.md", False),
        ("src/app.py", "src/**/*.py", True),
        ("src/pkg/app.py", "src/**/*.py", True),
        ("src/pkg/app.py", "src/*.py", False),
        ("cache/value.py", "**/*.py", True),
        ("value.py", "**/*.py", True),
        ("src/app.py", "tests/**/*.py", False),
    ],
)
def test_repository_pattern_semantics_are_rooted_and_segment_based(
    path: str, pattern: str, expected: bool
) -> None:
    """Repository globs have stable platform-independent matching semantics.

    Args:
        path:
            Repository path to match.
        pattern:
            Repository glob to apply.
        expected:
            Expected complete-match result.
    """

    assert matches_repository_pattern(path, pattern) is expected


def test_include_then_exclude_and_lexical_ordering() -> None:
    """Exclusions win over includes and output ordering ignores input ordering."""

    read_result = _read_result(
        _entry("src/z.py", b"z"),
        _entry("README.md", b"read me"),
        _entry("src/generated.py", b"generated"),
        _entry("src/a.py", b"a"),
        _entry("notes.txt", b"ignored\0binary"),
    )

    snapshot = build_repository_snapshot(
        "sample-project",
        read_result,
        _selection(
            include=("README.md", "src/**/*.py"),
            exclude=("src/generated.py",),
        ),
    )

    assert [file.path for file in snapshot.files] == [
        "README.md",
        "src/a.py",
        "src/z.py",
    ]


def test_text_and_content_hash_use_canonical_line_endings() -> None:
    """CRLF, lone CR, and absent final newlines have one canonical identity."""

    variants = (b"first\r\nsecond\r", b"first\nsecond", b"first\rsecond\n")
    snapshots = [
        build_repository_snapshot(
            "sample-project",
            _read_result(_entry("sample.py", content)),
            _selection(),
        )
        for content in variants
    ]

    assert {snapshot.files[0].text for snapshot in snapshots} == {"first\nsecond\n"}
    assert len({snapshot.files[0].content_hash for snapshot in snapshots}) == 1
    assert len({snapshot.source_fingerprint for snapshot in snapshots}) == 1
    assert snapshots[0].files[0].byte_size == len(b"first\nsecond\n")


@pytest.mark.parametrize(
    ("entry", "selection", "reason"),
    [
        (_entry("link.py", b"target", mode="120000"), _selection(), "symlink"),
        (
            _entry("directory.py", b"", mode="040000"),
            _selection(),
            "unsupported-entry",
        ),
        (
            _entry("module.py", None, mode="160000", object_type="commit"),
            _selection(),
            "submodule",
        ),
        (_entry("binary.py", b"a\0b"), _selection(), "binary-content"),
        (_entry("control.py", b"a\x01b"), _selection(), "binary-content"),
        (_entry("encoded.py", b"\xff"), _selection(), "invalid-utf8"),
        (
            _entry("large.py", b"12345"),
            _selection(max_file_bytes=4),
            "file-too-large",
        ),
        (
            _entry(
                "asset.py",
                b"version https://git-lfs.github.com/spec/v1\n"
                b"oid sha256:0123456789abcdef0123456789abcdef"
                b"0123456789abcdef0123456789abcdef\nsize 123\n",
            ),
            _selection(),
            "git-lfs-pointer",
        ),
    ],
    ids=[
        "symlink",
        "directory",
        "submodule",
        "binary-nul",
        "binary-control",
        "utf8",
        "oversized",
        "lfs",
    ],
)
def test_selected_unsupported_entries_are_rejected(
    entry: CommittedRepositoryEntry,
    selection: SelectionConfig,
    reason: str,
) -> None:
    """Every selected unsupported file shape fails with a typed reason.

    Args:
        entry:
            Unsupported committed entry.
        selection:
            Settings that select the entry.
        reason:
            Expected stable rejection reason.
    """

    with pytest.raises(SnapshotSelectionError) as caught:
        build_repository_snapshot("sample-project", _read_result(entry), selection)

    assert caught.value.path == entry.path
    assert caught.value.reason == reason


@pytest.mark.parametrize(
    "path", ["../secret.py", "src/../secret.py", "/etc/x.py", "C:/secret.py"]
)
def test_unsafe_reader_paths_are_rejected(path: str) -> None:
    """A provider cannot pass traversing or absolute paths into a snapshot.

    Args:
        path:
            Unsafe provider-supplied path.
    """

    with pytest.raises(SnapshotSelectionError) as caught:
        build_repository_snapshot(
            "sample-project", _read_result(_entry(path)), _selection()
        )

    assert caught.value.reason == "unsafe-path"


def test_fingerprint_ignores_repository_location_metadata_and_input_order() -> None:
    """Equivalent selected content has one identity across reader locations."""

    first = _read_result(
        _entry("src/b.py", b"b\n"),
        _entry("src/a.py", b"a\n"),
        repository_url="https://example.test/one",
    )
    second = RepositoryReadResult(
        repository_url="https://example.test/two",
        requested_ref="release",
        resolved_commit_sha="c" * 40,
        entries=tuple(reversed(first.entries)),
    )

    first_snapshot = build_repository_snapshot("one", first, _selection())
    second_snapshot = build_repository_snapshot("two", second, _selection())

    assert first_snapshot.source_fingerprint == second_snapshot.source_fingerprint
    assert first_snapshot.files == second_snapshot.files


def test_content_and_selection_rules_affect_source_fingerprint() -> None:
    """Source identity changes for content or effective rule configuration."""

    result = _read_result(_entry("src/a.py", b"a\n"))
    baseline = build_repository_snapshot("sample-project", result, _selection())
    changed_content = build_repository_snapshot(
        "sample-project",
        _read_result(_entry("src/a.py", b"changed\n")),
        _selection(),
    )
    changed_rules = build_repository_snapshot(
        "sample-project",
        result,
        _selection(include=("src/**/*.py",)),
    )

    assert baseline.source_fingerprint != changed_content.source_fingerprint
    assert baseline.source_fingerprint != changed_rules.source_fingerprint


def test_excluded_changes_do_not_affect_snapshot() -> None:
    """Changes to excluded committed files cannot alter selected source identity."""

    selection = _selection(exclude=("src/excluded.py",))
    baseline = build_repository_snapshot(
        "sample-project",
        _read_result(
            _entry("src/included.py", b"stable\n"),
            _entry("src/excluded.py", b"first\n"),
        ),
        selection,
    )
    changed = build_repository_snapshot(
        "sample-project",
        _read_result(
            _entry("src/included.py", b"stable\n"),
            _entry("src/excluded.py", b"second\0binary"),
        ),
        selection,
    )

    assert baseline == changed


def test_duplicate_provider_paths_are_rejected() -> None:
    """Ambiguous duplicate reader entries cannot enter canonical selection."""

    duplicate = _entry("src/app.py")

    with pytest.raises(SnapshotSelectionError) as caught:
        build_repository_snapshot(
            "sample-project", _read_result(duplicate, duplicate), _selection()
        )

    assert caught.value.reason == "duplicate-path"


def test_snapshot_fingerprint_does_not_contain_absolute_paths(tmp_path: Path) -> None:
    """The snapshot identity is independent of an arbitrary filesystem path.

    Args:
        tmp_path:
            Temporary path used only to make the forbidden value explicit.
    """

    snapshot = build_repository_snapshot(
        "sample-project",
        _read_result(_entry("src/app.py"), repository_url=str(tmp_path)),
        _selection(),
    )

    assert str(tmp_path) not in snapshot.source_fingerprint
