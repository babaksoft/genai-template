"""Read-only repository readers for portfolio snapshot workflows."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Literal

from genai_template.workflow.portfolio.config.models import LocalGitRepositoryConfig
from genai_template.workflow.portfolio.domain.errors import RepositoryReadError
from genai_template.workflow.portfolio.domain.snapshot import (
    CommittedRepositoryEntry,
    RepositoryReadResult,
)

_TreeRecord = tuple[str, Literal["blob", "commit"], str, str]


class LocalGitSnapshotReader:
    """Read committed objects from a local Git work tree without changing it."""

    def read(self, repository: LocalGitRepositoryConfig) -> RepositoryReadResult:
        """Resolve a ref and read all tracked entries from its commit.

        Git object commands are used exclusively; files in the work tree are never
        opened. Lazy fetching is disabled so a partial clone cannot turn this local
        operation into network access.

        Args:
            repository:
                Validated local-Git source and ref settings.

        Returns:
            Immutable resolved commit identity and committed entries.

        Raises:
            RepositoryReadError:
                If the path is missing, is not a Git work tree, the ref does not
                resolve to a commit, or committed objects cannot be read.
        """

        repository_path = repository.path
        self._validate_repository_path(repository_path, repository.ref)
        self._validate_work_tree(repository_path, repository.ref)

        commit_sha = self._resolve_commit(repository_path, repository.ref)
        repository_url = self._repository_url(repository_path, repository)
        entries = self._read_entries(repository_path, repository.ref, commit_sha)
        return RepositoryReadResult(
            repository_url=repository_url,
            requested_ref=repository.ref,
            resolved_commit_sha=commit_sha,
            entries=entries,
        )

    @staticmethod
    def _validate_repository_path(repository_path: Path, requested_ref: str) -> None:
        """Validate that a configured repository path is an existing directory.

        Args:
            repository_path:
                Configured local repository path.
            requested_ref:
                Ref included in contextual errors.

        Raises:
            RepositoryReadError:
                If the path does not exist or is not a directory.
        """

        if not repository_path.exists():
            raise RepositoryReadError(
                f"Repository path does not exist: {repository_path}",
                repository_path=repository_path,
                requested_ref=requested_ref,
                operation="validate repository path",
            )
        if not repository_path.is_dir():
            raise RepositoryReadError(
                f"Repository path is not a directory: {repository_path}",
                repository_path=repository_path,
                requested_ref=requested_ref,
                operation="validate repository path",
            )

    def _validate_work_tree(self, repository_path: Path, requested_ref: str) -> None:
        """Validate that a path belongs to a non-bare Git work tree.

        Args:
            repository_path:
                Existing directory to validate.
            requested_ref:
                Ref included in contextual errors.

        Raises:
            RepositoryReadError:
                If Git cannot inspect the path or it is not inside a work tree.
        """

        result = self._run_git(
            repository_path,
            ("rev-parse", "--is-inside-work-tree"),
            requested_ref=requested_ref,
            operation="validate Git work tree",
        )
        if result.stdout.strip() != b"true":
            raise RepositoryReadError(
                f"Path is not a Git work tree: {repository_path}",
                repository_path=repository_path,
                requested_ref=requested_ref,
                operation="validate Git work tree",
            )

    def _resolve_commit(self, repository_path: Path, requested_ref: str) -> str:
        """Resolve a configured ref to a full commit object identifier.

        Args:
            repository_path:
                Valid Git work-tree path.
            requested_ref:
                Ref to peel to a commit.

        Returns:
            Full lowercase hexadecimal commit object identifier.

        Raises:
            RepositoryReadError:
                If the ref is missing, malformed, or does not identify a commit.
        """

        result = self._run_git(
            repository_path,
            (
                "rev-parse",
                "--verify",
                "--end-of-options",
                f"{requested_ref}^{{commit}}",
            ),
            requested_ref=requested_ref,
            operation="resolve commit ref",
        )
        try:
            commit_sha = result.stdout.decode("ascii").strip().lower()
        except UnicodeDecodeError as error:
            raise RepositoryReadError(
                f"Git returned an invalid commit identifier for ref {requested_ref!r} "
                f"in {repository_path}",
                repository_path=repository_path,
                requested_ref=requested_ref,
                operation="resolve commit ref",
            ) from error
        if len(commit_sha) not in {40, 64} or any(
            character not in "0123456789abcdef" for character in commit_sha
        ):
            raise RepositoryReadError(
                f"Git returned an invalid commit identifier for ref {requested_ref!r} "
                f"in {repository_path}",
                repository_path=repository_path,
                requested_ref=requested_ref,
                operation="resolve commit ref",
            )
        return commit_sha

    def _repository_url(
        self,
        repository_path: Path,
        repository: LocalGitRepositoryConfig,
    ) -> str | None:
        """Return the explicit URL or the repository's optional origin URL.

        Normalization removes surrounding whitespace, trailing slashes, and one
        conventional ``.git`` suffix. Other URL components and local-path origins
        are preserved.

        Args:
            repository_path:
                Valid Git work-tree path.
            repository:
                Validated local-Git settings, including an optional URL override.

        Returns:
            Normalized explicit or origin URL, or null when no origin is configured.

        Raises:
            RepositoryReadError:
                If Git cannot inspect origin configuration.
        """

        if repository.url is not None:
            return self._normalize_repository_url_for_repository(
                repository.url, repository_path, repository.ref
            )

        result = self._run_git(
            repository_path,
            ("config", "--get", "remote.origin.url"),
            requested_ref=repository.ref,
            operation="read origin URL",
            accepted_returncodes=(0, 1),
        )
        if result.returncode == 1:
            return None
        try:
            origin_url = result.stdout.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RepositoryReadError(
                f"Git origin URL is not valid UTF-8 in {repository_path}",
                repository_path=repository_path,
                requested_ref=repository.ref,
                operation="read origin URL",
            ) from error
        return self._normalize_repository_url_for_repository(
            origin_url, repository_path, repository.ref
        )

    @classmethod
    def _normalize_repository_url_for_repository(
        cls,
        url: str,
        repository_path: Path,
        requested_ref: str,
    ) -> str:
        """Normalize a URL and attach repository context to invalid values.

        Args:
            url:
                Explicit or Git-configured repository URL.
            repository_path:
                Repository from which URL metadata was obtained.
            requested_ref:
                Ref used for the repository read operation.

        Returns:
            Normalized non-empty repository URL.

        Raises:
            RepositoryReadError:
                If normalization produces an empty URL.
        """

        try:
            return cls._normalize_repository_url(url)
        except ValueError as error:
            raise RepositoryReadError(
                f"Repository URL is empty in {repository_path}",
                repository_path=repository_path,
                requested_ref=requested_ref,
                operation="read origin URL",
            ) from error

    @staticmethod
    def _normalize_repository_url(url: str) -> str:
        """Normalize a configured Git repository URL for stable metadata.

        Args:
            url:
                Explicit or Git-configured repository URL.

        Returns:
            URL without surrounding whitespace, trailing slashes, or a final
            conventional ``.git`` suffix.

        Raises:
            ValueError:
                If normalization produces an empty URL.
        """

        normalized = url.strip().rstrip("/")
        if normalized.endswith(".git"):
            normalized = normalized[:-4].rstrip("/")
        if not normalized:
            raise ValueError("repository URL must not be empty")
        return normalized

    def _read_entries(
        self,
        repository_path: Path,
        requested_ref: str,
        commit_sha: str,
    ) -> tuple[CommittedRepositoryEntry, ...]:
        """Enumerate entries and load blob bytes from a resolved commit.

        Args:
            repository_path:
                Valid Git work-tree path.
            requested_ref:
                Ref used to resolve the commit, for contextual errors.
            commit_sha:
                Full resolved commit object identifier.

        Returns:
            Committed blob and gitlink entries ordered by decoded path.

        Raises:
            RepositoryReadError:
                If tree output is malformed or an object cannot be read.
        """

        result = self._run_git(
            repository_path,
            ("ls-tree", "-r", "-z", "--full-tree", commit_sha),
            requested_ref=requested_ref,
            operation="enumerate committed entries",
        )
        records = self._parse_tree_records(
            result.stdout,
            repository_path=repository_path,
            requested_ref=requested_ref,
        )
        blob_contents: dict[str, bytes] = {}
        entries: list[CommittedRepositoryEntry] = []
        for mode, object_type, object_id, path in records:
            content = None
            if object_type == "blob":
                if object_id not in blob_contents:
                    blob_contents[object_id] = self._run_git(
                        repository_path,
                        ("cat-file", "blob", object_id),
                        requested_ref=requested_ref,
                        operation=f"read committed blob {path!r}",
                    ).stdout
                content = blob_contents[object_id]
            entries.append(
                CommittedRepositoryEntry(
                    path=path,
                    mode=mode,
                    object_type=object_type,
                    object_id=object_id,
                    content=content,
                )
            )
        return tuple(sorted(entries, key=lambda entry: entry.path))

    @staticmethod
    def _parse_tree_records(
        output: bytes,
        *,
        repository_path: Path,
        requested_ref: str,
    ) -> list[_TreeRecord]:
        """Parse NUL-delimited ``git ls-tree`` output without path ambiguity.

        Args:
            output:
                Raw NUL-delimited Git tree output.
            repository_path:
                Repository path included in contextual errors.
            requested_ref:
                Requested ref included in contextual errors.

        Returns:
            Parsed mode, object type, object ID, and UTF-8 path records.

        Raises:
            RepositoryReadError:
                If Git returns malformed metadata or a non-UTF-8 path.
        """

        if not output:
            return []
        if not output.endswith(b"\0"):
            raise LocalGitSnapshotReader._malformed_tree_error(
                repository_path, requested_ref
            )

        parsed: list[_TreeRecord] = []
        for record in output[:-1].split(b"\0"):
            try:
                metadata, encoded_path = record.split(b"\t", 1)
                mode_bytes, object_type_bytes, object_id_bytes = metadata.split(b" ")
                mode = mode_bytes.decode("ascii")
                object_type = object_type_bytes.decode("ascii")
                object_id = object_id_bytes.decode("ascii").lower()
                path = encoded_path.decode("utf-8")
            except (UnicodeDecodeError, ValueError) as error:
                raise LocalGitSnapshotReader._malformed_tree_error(
                    repository_path, requested_ref
                ) from error
            if (
                len(mode) != 6
                or any(character not in "01234567" for character in mode)
                or len(object_id) not in {40, 64}
                or any(character not in "0123456789abcdef" for character in object_id)
                or not path
            ):
                raise LocalGitSnapshotReader._malformed_tree_error(
                    repository_path, requested_ref
                )
            if object_type == "blob":
                supported_object_type: Literal["blob", "commit"] = "blob"
            elif object_type == "commit":
                supported_object_type = "commit"
            else:
                raise LocalGitSnapshotReader._malformed_tree_error(
                    repository_path, requested_ref
                )
            parsed.append((mode, supported_object_type, object_id, path))
        return parsed

    @staticmethod
    def _malformed_tree_error(
        repository_path: Path, requested_ref: str
    ) -> RepositoryReadError:
        """Build a contextual malformed-tree exception.

        Args:
            repository_path:
                Repository that returned malformed tree output.
            requested_ref:
                Ref used for the read operation.

        Returns:
            Focused repository read error.
        """

        return RepositoryReadError(
            f"Git returned malformed tree data for ref {requested_ref!r} "
            f"in {repository_path}",
            repository_path=repository_path,
            requested_ref=requested_ref,
            operation="enumerate committed entries",
        )

    @staticmethod
    def _run_git(
        repository_path: Path,
        arguments: tuple[str, ...],
        *,
        requested_ref: str,
        operation: str,
        accepted_returncodes: tuple[int, ...] = (0,),
    ) -> subprocess.CompletedProcess[bytes]:
        """Run one read-only Git command and translate process failures.

        Args:
            repository_path:
                Working directory passed to Git with ``-C``.
            arguments:
                Git subcommand and arguments, passed without a shell.
            requested_ref:
                Ref included in contextual errors.
            operation:
                Description included in contextual errors.
            accepted_returncodes:
                Process return codes treated as expected outcomes.

        Returns:
            Completed Git process with raw standard output and error streams.

        Raises:
            RepositoryReadError:
                If Git cannot start or exits with an unexpected status.
        """

        environment = os.environ.copy()
        environment.update(
            {
                "GIT_NO_LAZY_FETCH": "1",
                "GIT_OPTIONAL_LOCKS": "0",
                "GIT_TERMINAL_PROMPT": "0",
            }
        )
        command = (
            "git",
            "--no-replace-objects",
            "-C",
            str(repository_path),
            *arguments,
        )
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                env=environment,
            )
        except OSError as error:
            raise RepositoryReadError(
                f"Unable to run Git while attempting to {operation} in "
                f"{repository_path}: {error}",
                repository_path=repository_path,
                requested_ref=requested_ref,
                operation=operation,
            ) from error

        if result.returncode not in accepted_returncodes:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            suffix = f": {detail}" if detail else ""
            raise RepositoryReadError(
                f"Unable to {operation} for ref {requested_ref!r} in "
                f"{repository_path}{suffix}",
                repository_path=repository_path,
                requested_ref=requested_ref,
                operation=operation,
            )
        return result
