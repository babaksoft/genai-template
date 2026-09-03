"""Unit tests for the corpus source service."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from genai_template.db.base import Base
from genai_template.db.models import Source
from genai_template.pipelines import IndexingPipeline
from genai_template.schemas import IndexingResult
from genai_template.services import SourceService


@pytest.fixture
def session_factory() -> sessionmaker[Session]:
    """Create an in-memory database session factory.

    Returns:
        Session factory with all project tables created.
    """

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    return sessionmaker(bind=engine, expire_on_commit=False)


def test_list_candidates_returns_uningested_immediate_directories(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
) -> None:
    """The service should expose only unregistered immediate directories."""

    (tmp_path / "zeta").mkdir()
    (tmp_path / "alpha").mkdir()

    with session_factory() as session:
        session.add(
            Source(
                name="zeta",
                directory=str((tmp_path / "zeta").resolve()),
                collection_name="source-zeta",
                documents_indexed=1,
                chunks_indexed=1,
                indexing_time=0.1,
            )
        )
        session.commit()

    service = SourceService(
        session_factory=session_factory,
        corpora_dir=tmp_path,
    )

    assert service.list_candidates() == ["alpha"]


def test_ingest_creates_persisted_source(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The service should index and persist a source for a candidate directory."""

    (tmp_path / "product-docs").mkdir()
    pipeline = MagicMock(spec=IndexingPipeline)
    pipeline.run.return_value = IndexingResult(
        documents_indexed=3,
        chunks_indexed=12,
        indexing_time=0.4,
    )
    service = SourceService(
        session_factory=session_factory,
        corpora_dir=tmp_path,
    )
    monkeypatch.setattr(
        service,
        "_create_indexing_pipeline",
        MagicMock(return_value=pipeline),
    )

    source = service.ingest("product-docs")

    assert source.id is not None
    assert source.name == "product-docs"
    assert source.directory == str((tmp_path / "product-docs").resolve())
    assert source.collection_name.startswith("source-")
    assert source.chunks_indexed == 12
    pipeline.run.assert_called_once_with((tmp_path / "product-docs").resolve())

    with session_factory() as session:
        persisted_source = session.get(Source, source.id)

    assert persisted_source is not None
    assert persisted_source.documents_indexed == 3
    assert persisted_source.chunks_indexed == 12


def test_ingest_rejects_paths_outside_corpus_root(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
) -> None:
    """The service should reject directory traversal attempts."""

    service = SourceService(
        session_factory=session_factory,
        corpora_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="immediate child"):
        service.ingest("../outside")


def test_ingest_rejects_duplicate_source_name(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
) -> None:
    """The service should preserve the one-source-per-directory-name rule."""

    (tmp_path / "product-docs").mkdir()
    pipeline = MagicMock(spec=IndexingPipeline)
    pipeline.run.return_value = IndexingResult(
        documents_indexed=1,
        chunks_indexed=2,
        indexing_time=0.1,
    )
    service = SourceService(
        session_factory=session_factory,
        corpora_dir=tmp_path,
    )
    service.ingest("product-docs")

    with pytest.raises(ValueError, match="already exists"):
        service.ingest("product-docs")


def test_refresh_replaces_source_collection_and_metadata(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refreshing a source should switch to a newly rebuilt collection."""

    corpus_directory = tmp_path / "product-docs"
    corpus_directory.mkdir()
    source = Source(
        name="product-docs",
        directory=str(corpus_directory.resolve()),
        collection_name="source-original",
        documents_indexed=1,
        chunks_indexed=2,
        indexing_time=0.1,
    )
    with session_factory() as session:
        session.add(source)
        session.commit()
        session.refresh(source)

    pipeline = MagicMock(spec=IndexingPipeline)
    pipeline.run.return_value = IndexingResult(
        documents_indexed=3,
        chunks_indexed=9,
        indexing_time=0.4,
    )
    delete_collection = MagicMock()
    service = SourceService(
        session_factory=session_factory,
        corpora_dir=tmp_path,
    )
    monkeypatch.setattr(
        service,
        "_create_indexing_pipeline",
        MagicMock(return_value=pipeline),
    )
    monkeypatch.setattr(service, "_delete_collection", delete_collection)

    refreshed = service.refresh(source.id)

    assert refreshed.collection_name != "source-original"
    assert refreshed.documents_indexed == 3
    assert refreshed.chunks_indexed == 9
    assert refreshed.indexing_time == 0.4
    pipeline.run.assert_called_once_with(corpus_directory.resolve())
    delete_collection.assert_called_once_with("source-original")


def test_refresh_preserves_source_when_rebuild_fails(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed refresh must leave the currently active source unchanged."""

    corpus_directory = tmp_path / "product-docs"
    corpus_directory.mkdir()
    source = Source(
        name="product-docs",
        directory=str(corpus_directory.resolve()),
        collection_name="source-original",
        documents_indexed=1,
        chunks_indexed=2,
        indexing_time=0.1,
    )
    with session_factory() as session:
        session.add(source)
        session.commit()
        session.refresh(source)

    pipeline = MagicMock(spec=IndexingPipeline)
    pipeline.run.side_effect = RuntimeError("embedding failed")
    delete_collection = MagicMock()
    service = SourceService(
        session_factory=session_factory,
        corpora_dir=tmp_path,
    )
    monkeypatch.setattr(
        service,
        "_create_indexing_pipeline",
        MagicMock(return_value=pipeline),
    )
    monkeypatch.setattr(service, "_delete_collection", delete_collection)

    with pytest.raises(RuntimeError, match="embedding failed"):
        service.refresh(source.id)

    with session_factory() as session:
        persisted_source = session.get(Source, source.id)

    assert persisted_source is not None
    assert persisted_source.collection_name == "source-original"
    assert persisted_source.documents_indexed == 1
    assert persisted_source.chunks_indexed == 2
    replacement_collection_name = delete_collection.call_args.args[0]
    assert replacement_collection_name.startswith("source-")
    assert replacement_collection_name != "source-original"
