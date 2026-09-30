"""Tests for durable index-build lifecycle persistence."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from genai_template.db.base import Base
from genai_template.db.models import IndexBuild, RagConfig, Source
from genai_template.services import IndexBuildService, IndexBuildTransitionError


@pytest.fixture
def session_factory() -> sessionmaker[Session]:
    """Create an in-memory database session factory.

    Returns:
        Session factory with all project tables created.
    """

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def identities(session_factory: sessionmaker[Session]) -> tuple[int, int]:
    """Persist one source and configuration for build tests.

    Args:
        session_factory:
            Database session factory.

    Returns:
        Source and RAG configuration identifiers.
    """

    with session_factory() as session:
        source = Source(name="docs", directory="/corpora/docs")
        config = RagConfig(config_fingerprint="a" * 64, config_json="{}")
        session.add_all((source, config))
        session.commit()
        return source.id, config.id


def start_build(
    service: IndexBuildService,
    identities: tuple[int, int],
    *,
    corpus_fingerprint: str | None = None,
) -> IndexBuild:
    """Start one consistently identified test build.

    Args:
        service:
            Build persistence service.
        identities:
            Source and configuration identifiers.
        corpus_fingerprint:
            Optional manifest identity.

    Returns:
        Newly persisted build model.
    """

    source_id, rag_config_id = identities
    return service.start(
        source_id=source_id,
        rag_config_id=rag_config_id,
        collection_name="idx-" + "b" * 56,
        index_fingerprint="c" * 64,
        corpus_fingerprint=corpus_fingerprint,
    )


def test_build_transitions_to_success_once(
    session_factory: sessionmaker[Session], identities: tuple[int, int]
) -> None:
    """A building attempt should accept exactly one successful completion."""

    service = IndexBuildService(session_factory)
    build = start_build(service, identities, corpus_fingerprint="d" * 64)

    completed = service.succeed(
        build.id,
        document_count=3,
        chunk_count=8,
        indexing_duration=0.25,
    )

    assert completed.status == "succeeded"
    assert completed.finished_at is not None
    assert completed.document_count == 3
    assert completed.chunk_count == 8
    assert completed.indexing_duration == 0.25
    with pytest.raises(IndexBuildTransitionError, match="already succeeded"):
        service.fail(
            build.id,
            failure_code="late-failure",
            failure_detail="must not overwrite success",
        )


def test_failure_is_sanitized_bounded_and_terminal(
    session_factory: sessionmaker[Session], identities: tuple[int, int]
) -> None:
    """Failure persistence should redact secrets and reject completion rewrites."""

    service = IndexBuildService(session_factory)
    build = start_build(service, identities)
    failed = service.fail(
        build.id,
        failure_code="Provider Failure !!!",
        failure_detail=(
            "token=very-secret /home/operator/corpus/file.md\n" + "x" * 700
        ),
    )

    assert failed.status == "failed"
    assert failed.failure_code == "provider-failure"
    assert failed.failure_detail is not None
    assert "very-secret" not in failed.failure_detail
    assert "/home/operator" not in failed.failure_detail
    assert "\n" not in failed.failure_detail
    assert len(failed.failure_detail) == 512
    assert failed.document_count is None
    assert failed.chunk_count is None
    with pytest.raises(IndexBuildTransitionError, match="already failed"):
        service.succeed(
            build.id,
            document_count=1,
            chunk_count=1,
            indexing_duration=0.1,
        )


def test_latest_queries_distinguish_attempt_from_success(
    session_factory: sessionmaker[Session], identities: tuple[int, int]
) -> None:
    """Newest-attempt and newest-success queries should preserve audit history."""

    service = IndexBuildService(session_factory)
    first = start_build(service, identities)
    service.succeed(
        first.id,
        document_count=1,
        chunk_count=2,
        indexing_duration=0.1,
    )
    second = start_build(service, identities)
    service.fail(
        second.id,
        failure_code="delete",
        failure_detail="Collection deletion failed.",
    )
    source_id, _ = identities
    latest = service.latest_attempt(
        source_id=source_id,
        collection_name="idx-" + "b" * 56,
        index_fingerprint="c" * 64,
    )
    successful = service.latest_successful(
        source_id=source_id,
        collection_name="idx-" + "b" * 56,
        index_fingerprint="c" * 64,
    )

    assert latest is not None
    assert latest.id == second.id
    assert successful is not None
    assert successful.id == first.id
