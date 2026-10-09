"""Deterministic selection and normalization of committed repository files."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePosixPath

from genai_template.workflow.portfolio.domain import (
    CommittedRepositoryEntry,
    RepositoryReadResult,
    RepositorySnapshot,
    SnapshotFile,
    SnapshotSelectionError,
)
from genai_template.workflow.portfolio.domain.config import SelectionConfig
from genai_template.workflow.portfolio.infrastructure.repositories import (
    matches_repository_pattern,
)

_REGULAR_FILE_MODES = {"100644", "100755"}
_LFS_VERSION_LINE = "version https://git-lfs.github.com/spec/v1"
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:/")


def build_repository_snapshot(
    project_slug: str,
    read_result: RepositoryReadResult,
    selection: SelectionConfig,
) -> RepositorySnapshot:
    """Select and normalize committed entries into a canonical snapshot.

    Include rules are applied before exclude rules. Rejected formats outside the
    selected set are ignored, while every selected entry must be a regular UTF-8
    text blob within the configured original-byte limit. CRLF and lone CR line
    endings become LF; non-empty text missing a terminal LF receives one.

    Args:
        project_slug:
            Stable project identifier assigned to the resulting snapshot.
        read_result:
            Provider-neutral committed entries and repository identity.
        selection:
            Validated allowlist, exclusions, and maximum original byte size.

    Returns:
        Canonical snapshot with files sorted lexically by normalized path.

    Raises:
        SnapshotSelectionError:
            If an entry path is unsafe or duplicated, or a selected entry is not
            an accepted regular UTF-8 text file.
    """

    entries_by_path: dict[str, CommittedRepositoryEntry] = {}
    for entry in read_result.entries:
        _validate_entry_path(entry.path)
        if entry.path in entries_by_path:
            raise SnapshotSelectionError(
                f"Committed entry path occurs more than once: {entry.path!r}",
                path=entry.path,
                reason="duplicate-path",
            )
        entries_by_path[entry.path] = entry

    selected_entries = (
        entry
        for path, entry in entries_by_path.items()
        if _matches_any(path, selection.include)
        and not _matches_any(path, selection.exclude)
    )
    files = tuple(
        _normalize_selected_entry(entry, selection.max_file_bytes)
        for entry in sorted(selected_entries, key=lambda item: item.path)
    )
    source_fingerprint = _source_fingerprint(selection, files)
    return RepositorySnapshot(
        project_slug=project_slug,
        repository_url=read_result.repository_url,
        requested_ref=read_result.requested_ref,
        resolved_commit_sha=read_result.resolved_commit_sha,
        source_fingerprint=source_fingerprint,
        files=files,
    )


def _validate_entry_path(path: str) -> None:
    """Reject paths that are not normalized repository-relative POSIX paths.

    Args:
        path:
            Path supplied by a repository reader.

    Raises:
        SnapshotSelectionError:
            If the path is absolute, traversing, empty, or not normalized.
    """

    parts = path.split("/")
    if (
        not path
        or path.startswith("/")
        or _WINDOWS_ABSOLUTE_PATH.match(path)
        or "\\" in path
        or "\0" in path
        or any(part in {"", ".", ".."} for part in parts)
        or PurePosixPath(path).is_absolute()
        or PurePosixPath(path).as_posix() != path
    ):
        raise SnapshotSelectionError(
            f"Committed entry path is not a normalized relative POSIX path: {path!r}",
            path=path,
            reason="unsafe-path",
        )


def _matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    """Return whether a path matches at least one configured pattern.

    Args:
        path:
            Normalized repository-relative path.
        patterns:
            Validated repository glob patterns.

    Returns:
        Whether any pattern matches the path.
    """

    return any(matches_repository_pattern(path, pattern) for pattern in patterns)


def _normalize_selected_entry(
    entry: CommittedRepositoryEntry, max_file_bytes: int
) -> SnapshotFile:
    """Validate and normalize one selected committed entry.

    Args:
        entry:
            Selected committed repository entry.
        max_file_bytes:
            Maximum accepted size before text normalization.

    Returns:
        Normalized text snapshot file.

    Raises:
        SnapshotSelectionError:
            If the entry is unsupported, too large, binary, invalid UTF-8, or a
            Git LFS pointer.
    """

    if entry.object_type == "commit" or entry.mode == "160000":
        _reject(entry.path, "submodule", "Git submodules are not supported")
    if entry.mode == "120000":
        _reject(entry.path, "symlink", "Symbolic links are not supported")
    if entry.object_type != "blob" or entry.mode not in _REGULAR_FILE_MODES:
        _reject(entry.path, "unsupported-entry", "Entry is not a regular file")

    content = entry.content
    if content is None:
        _reject(entry.path, "missing-content", "Regular file has no blob content")
    assert content is not None
    if len(content) > max_file_bytes:
        _reject(
            entry.path,
            "file-too-large",
            f"File exceeds the {max_file_bytes}-byte selection limit",
        )
    if _contains_binary_controls(content):
        _reject(entry.path, "binary-content", "File contains binary control bytes")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SnapshotSelectionError(
            f"Selected file is not valid UTF-8: {entry.path!r}",
            path=entry.path,
            reason="invalid-utf8",
        ) from error

    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    if normalized_text and not normalized_text.endswith("\n"):
        normalized_text += "\n"
    if _is_git_lfs_pointer(normalized_text):
        _reject(entry.path, "git-lfs-pointer", "Git LFS pointers are not supported")

    normalized_bytes = normalized_text.encode("utf-8")
    return SnapshotFile(
        path=entry.path,
        text=normalized_text,
        content_hash=hashlib.sha256(normalized_bytes).hexdigest(),
        byte_size=len(normalized_bytes),
    )


def _reject(path: str, reason: str, message: str) -> None:
    """Raise a consistently formatted selection failure.

    Args:
        path:
            Rejected repository-relative path.
        reason:
            Stable short rejection reason.
        message:
            Human-readable rejection explanation.

    Raises:
        SnapshotSelectionError:
            Always, for the supplied path and reason.
    """

    raise SnapshotSelectionError(
        f"{message}: {path!r}",
        path=path,
        reason=reason,
    )


def _is_git_lfs_pointer(text: str) -> bool:
    """Recognize a Git LFS v1 pointer rather than its external object.

    Args:
        text:
            Normalized decoded file text.

    Returns:
        Whether the text has the required Git LFS pointer fields.
    """

    lines = text.splitlines()
    if not lines or lines[0] != _LFS_VERSION_LINE:
        return False
    oid_values = [
        line.removeprefix("oid sha256:")
        for line in lines[1:]
        if line.startswith("oid sha256:")
    ]
    valid_oid = any(
        len(value) == 64 and all(character in "0123456789abcdef" for character in value)
        for value in oid_values
    )
    valid_size = any(
        line.removeprefix("size ").isdigit()
        for line in lines[1:]
        if line.startswith("size ")
    )
    return valid_oid and valid_size


def _contains_binary_controls(content: bytes) -> bool:
    """Detect control bytes that cannot occur in accepted source text.

    Horizontal tab, LF, form feed, and CR are accepted text controls. Other C0
    controls and DEL mark a blob as binary even when it happens to decode as UTF-8.

    Args:
        content:
            Original committed blob bytes.

    Returns:
        Whether the content contains a binary control byte.
    """

    accepted_controls = {9, 10, 12, 13}
    return any(
        (byte_value < 32 and byte_value not in accepted_controls) or byte_value == 127
        for byte_value in content
    )


def _source_fingerprint(
    selection: SelectionConfig, files: tuple[SnapshotFile, ...]
) -> str:
    """Calculate the canonical identity of selection settings and files.

    Args:
        selection:
            Stable file-selection settings.
        files:
            Canonically ordered normalized files.

    Returns:
        Hexadecimal SHA-256 source fingerprint.
    """

    payload = {
        "files": [
            {"content_hash": file.content_hash, "path": file.path} for file in files
        ],
        "selection": {
            "exclude": sorted(set(selection.exclude)),
            "include": sorted(set(selection.include)),
            "max_file_bytes": selection.max_file_bytes,
        },
        "version": 1,
    }
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
