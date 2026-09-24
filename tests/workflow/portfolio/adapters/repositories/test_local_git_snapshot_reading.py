"""Tests for read-only local Git Portfolio snapshot behavior."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from genai_template.workflow.portfolio import (
    LocalGitRepositoryConfig,
    LocalGitSnapshotReader,
    RepositoryReader,
    RepositoryReadError,
    RepositoryReadResult,
)


def _git(
    repository_path: Path,
    *arguments: str,
    environment: dict[str, str] | None = None,
) -> bytes:
    """Run Git in a test repository and return raw standard output.

    Args:
        repository_path:
            Repository in which to run Git.
        *arguments:
            Git subcommand and its arguments.
        environment:
            Optional process environment override.

    Returns:
        Raw standard output from the successful command.
    """

    result = subprocess.run(
        ("git", "-C", str(repository_path), *arguments),
        check=True,
        capture_output=True,
        env=environment,
    )
    return result.stdout


def _repository_config(
    repository_path: Path,
    *,
    ref: str = "HEAD",
    url: str | None = None,
) -> LocalGitRepositoryConfig:
    """Build validated local-Git settings for a test repository.

    Args:
        repository_path:
            Local test repository path.
        ref:
            Git revision for the reader to resolve.
        url:
            Optional repository URL override.

    Returns:
        Validated local-Git configuration.
    """

    return LocalGitRepositoryConfig(
        type="local_git",
        path=repository_path,
        ref=ref,
        url=url,
    )


@pytest.fixture
def committed_repository(tmp_path: Path) -> tuple[Path, str]:
    """Create a repository containing one deterministic committed tree.

    Args:
        tmp_path:
            Pytest temporary directory.

    Returns:
        Repository path and full initial commit identifier.
    """

    repository_path = tmp_path / "repository"
    repository_path.mkdir()
    _git(repository_path, "init", "--initial-branch=main")
    (repository_path / "tracked.txt").write_bytes(b"committed contents\n")
    nested_path = repository_path / "folder"
    nested_path.mkdir()
    (nested_path / "file with spaces.txt").write_bytes(b"space-safe\x00bytes\n")
    _git(repository_path, "add", "--all")

    environment = os.environ.copy()
    environment.update(
        {
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
        }
    )
    _git(
        repository_path,
        "-c",
        "user.name=Portfolio Tests",
        "-c",
        "user.email=portfolio@example.test",
        "commit",
        "--message=initial",
        environment=environment,
    )
    commit_sha = _git(repository_path, "rev-parse", "HEAD").decode("ascii").strip()
    return repository_path, commit_sha


def test_resolves_head_branch_tag_and_explicit_commit(
    committed_repository: tuple[Path, str],
) -> None:
    """All supported ref forms resolve to the same immutable commit."""

    repository_path, commit_sha = committed_repository
    _git(repository_path, "branch", "snapshot-branch", commit_sha)
    _git(repository_path, "tag", "snapshot-tag", commit_sha)
    reader = LocalGitSnapshotReader()

    results = [
        reader.read(_repository_config(repository_path, ref=ref))
        for ref in ("HEAD", "snapshot-branch", "snapshot-tag", commit_sha)
    ]

    assert all(result.resolved_commit_sha == commit_sha for result in results)
    assert [entry.path for entry in results[0].entries] == [
        "folder/file with spaces.txt",
        "tracked.txt",
    ]
    assert {entry.path: entry.content for entry in results[0].entries} == {
        "folder/file with spaces.txt": b"space-safe\x00bytes\n",
        "tracked.txt": b"committed contents\n",
    }


def test_reads_committed_objects_without_observing_or_changing_work_tree(
    committed_repository: tuple[Path, str],
) -> None:
    """Staged, modified, and untracked files do not affect snapshot reads."""

    repository_path, _ = committed_repository
    reader = LocalGitSnapshotReader()
    original_result = reader.read(_repository_config(repository_path))

    tracked_path = repository_path / "tracked.txt"
    tracked_path.write_bytes(b"staged contents\n")
    _git(repository_path, "add", "tracked.txt")
    tracked_path.write_bytes(b"working tree contents\n")
    (repository_path / "untracked.txt").write_bytes(b"not committed\n")
    branch_before = _git(repository_path, "symbolic-ref", "--short", "HEAD")
    status_before = _git(repository_path, "status", "--porcelain=v1", "-z")

    changed_work_tree_result = reader.read(_repository_config(repository_path))

    branch_after = _git(repository_path, "symbolic-ref", "--short", "HEAD")
    status_after = _git(repository_path, "status", "--porcelain=v1", "-z")
    assert changed_work_tree_result == original_result
    assert branch_after == branch_before
    assert status_after == status_before


def test_origin_url_is_normalized_and_explicit_url_takes_precedence(
    committed_repository: tuple[Path, str],
) -> None:
    """Origin metadata is stable and can be overridden by configuration."""

    repository_path, _ = committed_repository
    _git(
        repository_path,
        "remote",
        "add",
        "origin",
        "git@example.test:owner/project.git/",
    )
    reader = LocalGitSnapshotReader()

    origin_result = reader.read(_repository_config(repository_path))
    explicit_result = reader.read(
        _repository_config(
            repository_path,
            url=" https://example.test/portfolio/project.git/ ",
        )
    )

    assert origin_result.repository_url == "git@example.test:owner/project"
    assert explicit_result.repository_url == "https://example.test/portfolio/project"


def test_absent_origin_has_null_repository_url(
    committed_repository: tuple[Path, str],
) -> None:
    """A local-only repository has a documented null repository URL."""

    repository_path, _ = committed_repository

    result = LocalGitSnapshotReader().read(_repository_config(repository_path))

    assert result.repository_url is None


@pytest.mark.parametrize("path_kind", ["missing", "file"])
def test_missing_or_non_directory_repository_path_is_rejected(
    tmp_path: Path, path_kind: str
) -> None:
    """Invalid repository paths fail before invoking Git.

    Args:
        tmp_path:
            Pytest temporary directory.
        path_kind:
            Invalid path shape to construct.
    """

    repository_path = tmp_path / path_kind
    if path_kind == "file":
        repository_path.write_text("not a repository", encoding="utf-8")

    with pytest.raises(RepositoryReadError, match="Repository path") as caught:
        LocalGitSnapshotReader().read(_repository_config(repository_path))

    assert caught.value.repository_path == repository_path
    assert caught.value.operation == "validate repository path"


def test_non_repository_is_rejected(tmp_path: Path) -> None:
    """An existing ordinary directory is not accepted as a Git work tree."""

    with pytest.raises(RepositoryReadError, match="validate Git work tree") as caught:
        LocalGitSnapshotReader().read(_repository_config(tmp_path))

    assert caught.value.repository_path == tmp_path
    assert caught.value.requested_ref == "HEAD"


@pytest.mark.parametrize("ref", ["missing-ref", "blob-tag"])
def test_missing_and_non_commit_refs_are_rejected(
    committed_repository: tuple[Path, str], ref: str
) -> None:
    """Refs that cannot peel to commits report their path and ref.

    Args:
        committed_repository:
            Test repository and initial commit identifier.
        ref:
            Missing or non-commit ref to inspect.
    """

    repository_path, _ = committed_repository
    if ref == "blob-tag":
        blob_id = (
            _git(repository_path, "rev-parse", "HEAD:tracked.txt").decode().strip()
        )
        _git(repository_path, "tag", ref, blob_id)

    with pytest.raises(RepositoryReadError, match=ref) as caught:
        LocalGitSnapshotReader().read(_repository_config(repository_path, ref=ref))

    assert caught.value.repository_path == repository_path
    assert caught.value.requested_ref == ref
    assert caught.value.operation == "resolve commit ref"


def test_reader_satisfies_provider_neutral_protocol(
    committed_repository: tuple[Path, str],
) -> None:
    """Downstream callers can depend on the repository-reader protocol."""

    repository_path, commit_sha = committed_repository
    reader: RepositoryReader = LocalGitSnapshotReader()

    result = reader.read(_repository_config(repository_path))

    assert isinstance(reader, RepositoryReader)
    assert isinstance(result, RepositoryReadResult)
    assert result.requested_ref == "HEAD"
    assert result.resolved_commit_sha == commit_sha
