"""Persistence operations for durable index-build attempts."""

from __future__ import annotations

import re
from collections.abc import Callable

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from genai_template.db.models import IndexBuild
from genai_template.schemas import IndexBuildStatus
from genai_template.utils.datetime import utc_now

_FAILURE_CODE_LIMIT = 64
_FAILURE_DETAIL_LIMIT = 512
_SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|authorization|password|secret|token)\b"
    r"(\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+"
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_URL_CREDENTIAL_PATTERN = re.compile(r"(https?://)[^/@\s]+@")
_ABSOLUTE_PATH_PATTERN = re.compile(r"(?<![\w:])/(?:[^\s/]+/)+[^\s,;]*")


class IndexBuildTransitionError(ValueError):
    """Raised when a build is missing or is no longer in a mutable state."""


class IndexBuildService:
    """Persist and query short-transaction index-build lifecycle changes."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        """Initialize the persistence service.

        Args:
            session_factory:
                Factory that creates database sessions.
        """

        self._session_factory = session_factory

    def start(
        self,
        *,
        source_id: int,
        rag_config_id: int,
        collection_name: str,
        index_fingerprint: str,
        corpus_fingerprint: str | None,
    ) -> IndexBuild:
        """Commit a new building attempt.

        Args:
            source_id:
                Registered source being rebuilt.
            rag_config_id:
                Persisted configuration requesting the build.
            collection_name:
                Deterministic vector collection name.
            index_fingerprint:
                Complete index-affecting configuration fingerprint.
            corpus_fingerprint:
                Pinned manifest fingerprint, or ``None`` for a generic source.

        Returns:
            Detached durable building attempt.
        """

        build = IndexBuild(
            source_id=source_id,
            rag_config_id=rag_config_id,
            collection_name=collection_name,
            index_fingerprint=index_fingerprint,
            corpus_fingerprint=corpus_fingerprint,
            status=IndexBuildStatus.BUILDING.value,
        )
        with self._session_factory() as session:
            session.add(build)
            session.commit()
            session.refresh(build)
            session.expunge(build)
        return build

    def succeed(
        self,
        build_id: int,
        *,
        document_count: int,
        chunk_count: int,
        indexing_duration: float,
    ) -> IndexBuild:
        """Commit successful completion of a building attempt.

        Args:
            build_id:
                Attempt to complete.
            document_count:
                Number of documents processed.
            chunk_count:
                Verified number of chunks stored.
            indexing_duration:
                Measured pipeline duration in seconds.

        Returns:
            Detached completed attempt.

        Raises:
            IndexBuildTransitionError:
                If the attempt is missing or already terminal.
            ValueError:
                If a count or duration is negative.
        """

        if document_count < 0 or chunk_count < 0 or indexing_duration < 0:
            raise ValueError("Successful build metrics cannot be negative.")

        with self._session_factory() as session:
            build = self._require_building(session, build_id)
            build.status = IndexBuildStatus.SUCCEEDED.value
            build.finished_at = utc_now()
            build.document_count = document_count
            build.chunk_count = chunk_count
            build.indexing_duration = indexing_duration
            build.failure_code = None
            build.failure_detail = None
            session.commit()
            session.refresh(build)
            session.expunge(build)
        return build

    def fail(
        self,
        build_id: int,
        *,
        failure_code: str,
        failure_detail: str,
    ) -> IndexBuild:
        """Commit sanitized failed completion of a building attempt.

        Args:
            build_id:
                Attempt to complete.
            failure_code:
                Stable machine-readable failure category.
            failure_detail:
                Operator-facing summary to sanitize and bound.

        Returns:
            Detached failed attempt.

        Raises:
            IndexBuildTransitionError:
                If the attempt is missing or already terminal.
        """

        code = self.sanitize_failure_code(failure_code)
        detail = self.sanitize_failure_detail(failure_detail)
        with self._session_factory() as session:
            build = self._require_building(session, build_id)
            build.status = IndexBuildStatus.FAILED.value
            build.finished_at = utc_now()
            build.failure_code = code
            build.failure_detail = detail
            build.document_count = None
            build.chunk_count = None
            build.indexing_duration = None
            session.commit()
            session.refresh(build)
            session.expunge(build)
        return build

    def latest_attempt(
        self,
        *,
        source_id: int,
        collection_name: str,
        index_fingerprint: str,
    ) -> IndexBuild | None:
        """Return the newest attempt for one deterministic source index.

        Args:
            source_id:
                Registered source identifier.
            collection_name:
                Deterministic vector collection name.
            index_fingerprint:
                Complete index-affecting configuration fingerprint.

        Returns:
            Detached newest attempt, or ``None`` when no attempt exists.
        """

        return self._latest(
            source_id=source_id,
            collection_name=collection_name,
            index_fingerprint=index_fingerprint,
        )

    def latest_successful(
        self,
        *,
        source_id: int,
        collection_name: str,
        index_fingerprint: str,
    ) -> IndexBuild | None:
        """Return the newest successful attempt for one deterministic index.

        Args:
            source_id:
                Registered source identifier.
            collection_name:
                Deterministic vector collection name.
            index_fingerprint:
                Complete index-affecting configuration fingerprint.

        Returns:
            Detached newest successful attempt, or ``None`` when absent.
        """

        return self._latest(
            source_id=source_id,
            collection_name=collection_name,
            index_fingerprint=index_fingerprint,
            status=IndexBuildStatus.SUCCEEDED,
        )

    @staticmethod
    def sanitize_failure_code(value: str) -> str:
        """Normalize and bound a machine-readable failure code.

        Args:
            value:
                Untrusted proposed failure code.

        Returns:
            Non-empty lowercase code containing safe identifier characters.
        """

        normalized = re.sub(r"[^a-z0-9_-]+", "-", value.strip().lower()).strip("-")
        return (normalized or "index-build-failed")[:_FAILURE_CODE_LIMIT]

    @staticmethod
    def sanitize_failure_detail(value: str) -> str:
        """Redact common sensitive values and bound an operator summary.

        Args:
            value:
                Untrusted proposed failure detail.

        Returns:
            Single-line bounded detail without common credentials or local paths.
        """

        detail = " ".join(value.split())
        detail = _SECRET_PATTERN.sub(r"\1\2[REDACTED]", detail)
        detail = _BEARER_PATTERN.sub("Bearer [REDACTED]", detail)
        detail = _URL_CREDENTIAL_PATTERN.sub(r"\1[REDACTED]@", detail)
        detail = _ABSOLUTE_PATH_PATTERN.sub("[LOCAL_PATH]", detail)
        if not detail:
            detail = "Index rebuild failed."
        return detail[:_FAILURE_DETAIL_LIMIT]

    def _latest(
        self,
        *,
        source_id: int,
        collection_name: str,
        index_fingerprint: str,
        status: IndexBuildStatus | None = None,
    ) -> IndexBuild | None:
        """Execute one newest-build query.

        Args:
            source_id:
                Registered source identifier.
            collection_name:
                Deterministic vector collection name.
            index_fingerprint:
                Complete index-affecting configuration fingerprint.
            status:
                Optional lifecycle state filter.

        Returns:
            Detached newest matching build, or ``None``.
        """

        statement: Select[tuple[IndexBuild]] = select(IndexBuild).where(
            IndexBuild.source_id == source_id,
            IndexBuild.collection_name == collection_name,
            IndexBuild.index_fingerprint == index_fingerprint,
        )
        if status is not None:
            statement = statement.where(IndexBuild.status == status.value)
        statement = statement.order_by(
            IndexBuild.started_at.desc(),
            IndexBuild.id.desc(),
        )
        with self._session_factory() as session:
            build = session.scalar(statement.limit(1))
            if build is not None:
                session.expunge(build)
            return build

    @staticmethod
    def _require_building(session: Session, build_id: int) -> IndexBuild:
        """Load a mutable attempt or reject the transition.

        Args:
            session:
                Active short-lived database session.
            build_id:
                Attempt identifier.

        Returns:
            Persistent building attempt.

        Raises:
            IndexBuildTransitionError:
                If the attempt is missing or already terminal.
        """

        build = session.get(IndexBuild, build_id)
        if build is None:
            raise IndexBuildTransitionError(f"Index build {build_id} does not exist.")
        if build.status != IndexBuildStatus.BUILDING.value:
            raise IndexBuildTransitionError(
                f"Index build {build_id} is already {build.status}."
            )
        return build
