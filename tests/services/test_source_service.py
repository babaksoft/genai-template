"""Unit tests for source registration and deterministic index lifecycle."""

from hashlib import sha256
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from genai_template.config import index_config_fingerprint, load_rag_config
from genai_template.db.base import Base
from genai_template.db.models import Source
from genai_template.pipelines import IndexingPipeline
from genai_template.schemas import IndexingResult
from genai_template.services import RagConfigService, SourceService


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
    pipeline.run.return_value = IndexingResult(
        documents_indexed=3, chunks_indexed=12, indexing_time=0.4
    )
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
    pipeline.run.assert_called_once_with(directory.resolve())
    assert result.documents_indexed == 3
    assert result.chunks_indexed == 12
    assert result.indexing_time == 0.4


def test_rebuild_lock_is_shared_for_the_same_collection() -> None:
    """Separate service instances should serialize the same collection."""

    first = SourceService._get_rebuild_lock("idx-shared")
    second = SourceService._get_rebuild_lock("idx-shared")
    other = SourceService._get_rebuild_lock("idx-other")

    assert first is second
    assert first is not other
