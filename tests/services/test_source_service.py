"""Unit tests for source registration and deterministic index lifecycle."""

from hashlib import sha256
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from genai_template.config import index_config_fingerprint, load_rag_config
from genai_template.db.base import Base
from genai_template.db.models import IndexBuild, Source
from genai_template.pipelines import IndexingPipeline
from genai_template.schemas import IndexingResult
from genai_template.services import (
    IndexBuildInProgressError,
    IndexCountMismatchError,
    RagConfigService,
    SourceService,
)


@pytest.fixture
def session_factory() -> sessionmaker[Session]:
    """Create an in-memory database session factory.

    Returns:
        Session factory with all project tables created.
    """

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def test_list_candidates_excludes_registered_directories(
    tmp_path: Path, session_factory: sessionmaker[Session]
) -> None:
    """Candidates should contain only unregistered immediate directories."""

    (tmp_path / "zeta").mkdir()
    (tmp_path / "alpha").mkdir()
    with session_factory() as session:
        session.add(Source(name="zeta", directory=str((tmp_path / "zeta").resolve())))
        session.commit()

    service = SourceService(session_factory, tmp_path)

    assert service.list_candidates() == ["alpha"]


def test_register_persists_only_source_directory(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registration should not construct or populate a vector index."""

    directory = tmp_path / "product-docs"
    directory.mkdir()
    create_store = MagicMock()
    monkeypatch.setattr(
        "genai_template.services.source_service.create_vector_store", create_store
    )
    service = SourceService(session_factory, tmp_path)

    source = service.register("product-docs")

    assert source.name == "product-docs"
    assert source.directory == str(directory.resolve())
    assert source.created_at is not None
    create_store.assert_not_called()
    with session_factory() as session:
        assert session.get(Source, source.id) is not None


def test_register_rejects_invalid_and_duplicate_sources(
    tmp_path: Path, session_factory: sessionmaker[Session]
) -> None:
    """Registration should retain source path and uniqueness validation."""

    (tmp_path / "docs").mkdir()
    service = SourceService(session_factory, tmp_path)
    service.register("docs")

    with pytest.raises(ValueError, match="already exists"):
        service.register("docs")
    with pytest.raises(ValueError, match="immediate child"):
        service.register("../outside")


def test_collection_name_is_fixed_length_and_deterministic() -> None:
    """Collection identity should hash the source ID and index fingerprint."""

    config = load_rag_config()
    identity = f"7:{index_config_fingerprint(config)}"
    expected = f"idx-{sha256(identity.encode('utf-8')).hexdigest()[:56]}"

    assert SourceService.index_collection_name(7, config) == expected
    assert len(expected) == 60
    assert SourceService.index_collection_name(8, config) != expected


def test_configs_with_shared_index_settings_share_collection() -> None:
    """Retrieval and generation changes should reuse the same source index."""

    config = load_rag_config()
    changed = config.model_copy(
        update={
            "retrieval": config.retrieval.model_copy(update={"top_k": 99}),
            "llm": config.llm.model_copy(update={"model_name": "another-model"}),
        }
    )

    assert SourceService.index_collection_name(
        4, config
    ) == SourceService.index_collection_name(4, changed)


def test_rebuild_loads_persisted_config_and_replaces_collection(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rebuild should use the selected config and return transient metrics."""

    directory = tmp_path / "docs"
    directory.mkdir()
    config = load_rag_config()
    config_registry = RagConfigService(session_factory)
    config_record = config_registry.register_config(config)
    service = SourceService(session_factory, tmp_path, config_registry)
    source = service.register("docs")
    store = MagicMock()
    pipeline = MagicMock(spec=IndexingPipeline)
    pipeline.run_loaded.return_value = IndexingResult(
        documents_indexed=3, chunks_indexed=12, indexing_time=0.4
    )
    store.count.return_value = 12
    create_store = MagicMock(return_value=store)
    create_pipeline = MagicMock(return_value=pipeline)
    monkeypatch.setattr(
        "genai_template.services.source_service.create_vector_store", create_store
    )
    monkeypatch.setattr(service, "_create_indexing_pipeline", create_pipeline)

    result = service.rebuild_index(source.id, config_record.id)

    collection = SourceService.index_collection_name(source.id, config)
    create_store.assert_called_once_with(config.vector_store, collection)
    store.delete.assert_called_once_with()
    create_pipeline.assert_called_once_with(config, store)
    pipeline.run_loaded.assert_called_once()
    assert result.documents_indexed == 3
    assert result.chunks_indexed == 12
    assert result.indexing_time == 0.4
    with session_factory() as session:
        build = session.query(IndexBuild).one()
        assert build.status == "succeeded"
        assert build.document_count == 3
        assert build.chunk_count == 12
        assert build.corpus_fingerprint is None


def test_rebuild_lock_is_shared_for_the_same_collection() -> None:
    """Separate service instances should serialize the same collection."""

    first = SourceService._get_rebuild_lock("idx-shared")
    second = SourceService._get_rebuild_lock("idx-shared")
    other = SourceService._get_rebuild_lock("idx-other")

    assert first is second
    assert first is not other


def test_rebuild_validation_failure_does_not_create_attempt(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure before destructive work should not create an audit attempt."""

    directory = tmp_path / "docs"
    directory.mkdir()
    registry = RagConfigService(session_factory)
    config_record = registry.register_config(load_rag_config())
    service = SourceService(session_factory, tmp_path, registry)
    source = service.register("docs")
    create_store = MagicMock()
    monkeypatch.setattr(
        service, "_load_source", MagicMock(side_effect=ValueError("bad"))
    )
    monkeypatch.setattr(
        "genai_template.services.source_service.create_vector_store", create_store
    )

    with pytest.raises(ValueError, match="bad"):
        service.rebuild_index(source.id, config_record.id)

    create_store.assert_not_called()
    with session_factory() as session:
        assert session.query(IndexBuild).count() == 0


def test_rebuild_delete_failure_is_persisted_and_lock_is_released(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A destructive-phase error should leave a failed row and release the lock."""

    directory = tmp_path / "docs"
    directory.mkdir()
    config = load_rag_config()
    registry = RagConfigService(session_factory)
    config_record = registry.register_config(config)
    service = SourceService(session_factory, tmp_path, registry)
    source = service.register("docs")
    store = MagicMock()

    def fail_after_asserting_durable_attempt() -> None:
        """Prove the building row committed before destructive work."""

        with session_factory() as session:
            assert session.query(IndexBuild).one().status == "building"
        raise RuntimeError("provider payload must not persist")

    store.delete.side_effect = fail_after_asserting_durable_attempt
    monkeypatch.setattr(
        "genai_template.services.source_service.create_vector_store",
        MagicMock(return_value=store),
    )

    with pytest.raises(RuntimeError, match="provider payload"):
        service.rebuild_index(source.id, config_record.id)

    collection = SourceService.index_collection_name(source.id, config)
    assert service._get_rebuild_lock(collection).acquire(blocking=False)
    service._get_rebuild_lock(collection).release()
    with session_factory() as session:
        build = session.query(IndexBuild).one()
        assert build.status == "failed"
        assert build.failure_code == "collection_delete"
        assert build.failure_detail is not None
        assert "provider payload" not in build.failure_detail


def test_rebuild_rejects_count_mismatch(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A collection count mismatch should prevent successful completion."""

    directory = tmp_path / "docs"
    directory.mkdir()
    registry = RagConfigService(session_factory)
    config_record = registry.register_config(load_rag_config())
    service = SourceService(session_factory, tmp_path, registry)
    source = service.register("docs")
    store = MagicMock()
    store.count.return_value = 1
    pipeline = MagicMock(spec=IndexingPipeline)
    pipeline.run_loaded.return_value = IndexingResult(
        documents_indexed=2,
        chunks_indexed=3,
        indexing_time=0.2,
    )
    monkeypatch.setattr(
        "genai_template.services.source_service.create_vector_store",
        MagicMock(return_value=store),
    )
    monkeypatch.setattr(
        service, "_create_indexing_pipeline", MagicMock(return_value=pipeline)
    )

    with pytest.raises(IndexCountMismatchError):
        service.rebuild_index(source.id, config_record.id)

    with session_factory() as session:
        build = session.query(IndexBuild).one()
        assert build.status == "failed"
        assert build.failure_code == "count_verification"


def test_rebuild_success_persistence_failure_marks_attempt_failed(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure to record success should conservatively terminalize the attempt."""

    directory = tmp_path / "docs"
    directory.mkdir()
    registry = RagConfigService(session_factory)
    config_record = registry.register_config(load_rag_config())
    service = SourceService(session_factory, tmp_path, registry)
    source = service.register("docs")
    store = MagicMock()
    store.count.return_value = 2
    pipeline = MagicMock(spec=IndexingPipeline)
    pipeline.run_loaded.return_value = IndexingResult(
        documents_indexed=1,
        chunks_indexed=2,
        indexing_time=0.1,
    )
    monkeypatch.setattr(
        "genai_template.services.source_service.create_vector_store",
        MagicMock(return_value=store),
    )
    monkeypatch.setattr(
        service, "_create_indexing_pipeline", MagicMock(return_value=pipeline)
    )
    monkeypatch.setattr(
        service._index_build_service,
        "succeed",
        MagicMock(side_effect=RuntimeError("database write failed")),
    )

    with pytest.raises(RuntimeError, match="database write failed"):
        service.rebuild_index(source.id, config_record.id)

    with session_factory() as session:
        build = session.query(IndexBuild).one()
        assert build.status == "failed"
        assert build.failure_code == "success_persistence"


def test_rebuild_lock_conflict_is_non_blocking_and_has_no_attempt(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
) -> None:
    """A process-local owner should produce an immediate focused conflict."""

    (tmp_path / "docs").mkdir()
    config = load_rag_config()
    registry = RagConfigService(session_factory)
    config_record = registry.register_config(config)
    service = SourceService(session_factory, tmp_path, registry)
    source = service.register("docs")
    collection = SourceService.index_collection_name(source.id, config)
    lock = service._get_rebuild_lock(collection)
    lock.acquire()
    try:
        with pytest.raises(IndexBuildInProgressError, match="already in progress"):
            service.rebuild_index(source.id, config_record.id)
    finally:
        lock.release()

    with session_factory() as session:
        assert session.query(IndexBuild).count() == 0
