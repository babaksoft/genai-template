"""Corpus source lifecycle service."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import ClassVar

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from genai_template.components.readers import PortfolioReader, TextReader
from genai_template.config import RagConfig, index_config_fingerprint
from genai_template.db.models import Source
from genai_template.factories import (
    create_embedder,
    create_splitter,
    create_vector_store,
)
from genai_template.observability import application_span
from genai_template.pipelines import IndexingPipeline
from genai_template.protocols import VectorStore
from genai_template.schemas import IndexBuildStatus, IndexingResult, LoadedDocuments
from genai_template.services.index_build_service import IndexBuildService
from genai_template.services.rag_config_service import RagConfigService

logger = logging.getLogger(__name__)


class IndexBuildInProgressError(RuntimeError):
    """Raised when this process is already rebuilding the selected index."""


class IndexCountMismatchError(RuntimeError):
    """Raised when a completed pipeline has not stored its reported chunks."""


class SourceService:
    """Register document corpora and manage their deterministic indexes."""

    _locks_guard: ClassVar[Lock] = Lock()
    _rebuild_locks: ClassVar[dict[str, Lock]] = {}

    def __init__(
        self,
        session_factory: Callable[[], Session],
        corpora_dir: Path,
        rag_config_service: RagConfigService | None = None,
        index_build_service: IndexBuildService | None = None,
    ) -> None:
        """Initialize the source service.

        Args:
            session_factory:
                Factory that creates database sessions.
            corpora_dir:
                Root directory containing one directory per corpus.
            rag_config_service:
                Registry used to load persisted RAG configurations. A registry
                backed by ``session_factory`` is constructed when omitted.
            index_build_service:
                Persistence service for durable build attempts. A service backed
                by ``session_factory`` is constructed when omitted.
        """

        self._session_factory = session_factory
        self._corpora_dir = corpora_dir.resolve()
        self._rag_config_service = rag_config_service or RagConfigService(
            session_factory
        )
        self._index_build_service = index_build_service or IndexBuildService(
            session_factory
        )

    def list_candidates(self) -> list[str]:
        """List immediate corpus directories available for registration.

        Returns:
            Sorted directory basenames under the configured corpus root.
        """

        if not self._corpora_dir.exists():
            return []

        with self._session_factory() as session:
            source_names = set(session.scalars(select(Source.name)))

        return sorted(
            path.name
            for path in self._corpora_dir.iterdir()
            if (
                path.is_dir()
                and path.resolve().parent == self._corpora_dir
                and path.name not in source_names
            )
        )

    def list_sources(self) -> list[Source]:
        """List all registered sources.

        Returns:
            Sources ordered by name.
        """

        with self._session_factory() as session:
            return list(session.scalars(select(Source).order_by(Source.name)))

    def get_source(self, source_id: int) -> Source:
        """Get a registered source by identifier.

        Args:
            source_id:
                Unique identifier of the requested source.

        Returns:
            Persisted source metadata.

        Raises:
            ValueError:
                If no source has the supplied identifier.
        """

        with self._session_factory() as session:
            source = session.get(Source, source_id)

        if source is None:
            raise ValueError(f"Source {source_id} does not exist.")

        return source

    def register(self, directory_name: str) -> Source:
        """Register one previously prepared corpus directory.

        Args:
            directory_name:
                Immediate child directory name under the configured corpus root.

        Returns:
            Persisted source.

        Raises:
            ValueError:
                If the directory name is outside the corpus root or already
                registered as a source.
            FileNotFoundError:
                If the requested directory does not exist.
            NotADirectoryError:
                If the requested path is not a directory.
        """

        directory = self._resolve_directory(directory_name)
        source_name = directory.name

        with self._session_factory() as session:
            existing_source = session.scalar(
                select(Source).where(Source.name == source_name)
            )
            if existing_source is not None:
                raise ValueError(f"Source '{source_name}' already exists.")

        source = Source(
            name=source_name,
            directory=str(directory),
        )

        try:
            with self._session_factory() as session:
                session.add(source)
                session.commit()
                session.refresh(source)
        except IntegrityError as exc:
            raise ValueError(f"Source '{source_name}' already exists.") from exc

        logger.info("Registered source %d from '%s'.", source.id, source.directory)
        return source

    def rebuild_index(self, source_id: int, rag_config_id: int) -> IndexingResult:
        """Rebuild the deterministic index for a source and configuration.

        Args:
            source_id:
                Identifier of the source to rebuild.
            rag_config_id:
                Identifier of the persisted configuration used for indexing.

        Returns:
            Counts and duration for this build.

        Raises:
            ValueError:
                If the source or RAG configuration does not exist, or if the
                persisted source directory is no longer valid.
            FileNotFoundError:
                If the source directory no longer exists.
            NotADirectoryError:
                If the source path is no longer a directory.
            IndexBuildInProgressError:
                If another request in this process owns the collection lock.
            IndexCountMismatchError:
                If the vector store count differs from the pipeline result.
        """

        source = self.get_source(source_id)
        directory = self._resolve_registered_directory(source)
        record = self._rag_config_service.get_config(rag_config_id)
        config = self._rag_config_service.parse_config(record)
        collection_name = self.index_collection_name(source_id, config)
        index_fingerprint = index_config_fingerprint(config)
        rebuild_lock = self._get_rebuild_lock(collection_name)

        if not rebuild_lock.acquire(blocking=False):
            raise IndexBuildInProgressError(
                f"Index rebuild already in progress for collection '{collection_name}'."
            )

        try:
            with application_span(
                "rag.index.rebuild",
                "CHAIN",
                {
                    "rag.source.id": source.id,
                    "rag.config.id": record.id,
                    "rag.config.fingerprint": record.config_fingerprint,
                    "rag.index.fingerprint": index_fingerprint,
                    "rag.index.collection": collection_name,
                    "rag.splitter.type": config.splitter.type,
                    "rag.embedding.provider": config.embedder.type,
                    "rag.embedding.model": config.embedder.model_name,
                    "rag.vector_store.type": config.vector_store.type,
                    "rag.vector_store.distance": config.vector_store.distance.value,
                },
            ) as span:
                with application_span("rag.index.load", "CHAIN") as load_span:
                    loaded = self._load_source(directory)
                    load_span.set_attribute("rag.document.count", len(loaded.documents))
                corpus_fingerprint = self._corpus_fingerprint(loaded)
                build = self._index_build_service.start(
                    source_id=source.id,
                    rag_config_id=record.id,
                    collection_name=collection_name,
                    index_fingerprint=index_fingerprint,
                    corpus_fingerprint=corpus_fingerprint,
                )
                span.set_attribute("rag.index.build.id", build.id)
                span.set_attribute(
                    "rag.index.build.status", IndexBuildStatus.BUILDING.value
                )
                if corpus_fingerprint is not None:
                    span.set_attribute("rag.corpus.fingerprint", corpus_fingerprint)

                phase = "store_initialization"
                try:
                    store = create_vector_store(config.vector_store, collection_name)

                    phase = "collection_delete"
                    with application_span(
                        "rag.index.delete",
                        "CHAIN",
                        {"rag.index.collection": collection_name},
                    ):
                        store.delete()

                    phase = "indexing"
                    pipeline = self._create_indexing_pipeline(config, store)
                    result = pipeline.run_loaded(loaded)

                    phase = "count_verification"
                    stored_count = store.count()
                    if stored_count != result.chunks_indexed:
                        raise IndexCountMismatchError(
                            "Vector store count does not match the indexed chunk count."
                        )

                    phase = "success_persistence"
                    completed = self._index_build_service.succeed(
                        build.id,
                        document_count=result.documents_indexed,
                        chunk_count=stored_count,
                        indexing_duration=result.indexing_time,
                    )
                except Exception as exc:
                    self._record_build_failure(build.id, phase, exc)
                    span.set_attribute(
                        "rag.index.build.status", IndexBuildStatus.FAILED.value
                    )
                    span.set_attribute("rag.index.build.failure_code", phase)
                    raise

                span.set_attribute(
                    "rag.index.build.status", IndexBuildStatus.SUCCEEDED.value
                )
                span.set_attribute("rag.document.count", completed.document_count or 0)
                span.set_attribute("rag.chunk.count", completed.chunk_count or 0)
                span.set_attribute(
                    "rag.index.duration", completed.indexing_duration or 0.0
                )
        finally:
            rebuild_lock.release()

        logger.info(
            "Rebuilt source %d index '%s' with RAG config %d: documents=%d, "
            "chunks=%d, duration=%.3f second(s).",
            source_id,
            collection_name,
            rag_config_id,
            result.documents_indexed,
            result.chunks_indexed,
            result.indexing_time,
        )
        return result

    @staticmethod
    def _load_source(directory: Path) -> LoadedDocuments:
        """Validate and load a source before a destructive attempt begins.

        Args:
            directory:
                Registered generic directory or Portfolio publication pointer.

        Returns:
            In-memory documents with pinned provenance when manifest-backed.
        """

        reader = (
            PortfolioReader()
            if (directory / "manifest.json").is_file()
            else TextReader()
        )
        return reader.load(directory)

    @staticmethod
    def _corpus_fingerprint(loaded: LoadedDocuments) -> str | None:
        """Return the pinned manifest fingerprint when present.

        Args:
            loaded:
                Validated loaded documents.

        Returns:
            Corpus fingerprint, or ``None`` for a generic source.
        """

        if loaded.provenance is None:
            return None

        return loaded.provenance.corpus_fingerprint

    def _record_build_failure(
        self,
        build_id: int,
        phase: str,
        error: Exception,
    ) -> None:
        """Best-effort persist a safe terminal failure summary.

        Args:
            build_id:
                Durable attempt that failed.
            phase:
                Stable lifecycle phase in which the failure occurred.
            error:
                Original error, used only for its exception class name.
        """

        detail = f"Index rebuild failed during {phase} ({type(error).__name__})."
        try:
            self._index_build_service.fail(
                build_id,
                failure_code=phase,
                failure_detail=detail,
            )
        except Exception:
            logger.exception(
                "Could not persist failure for index build %d; it remains conservatively unavailable.",
                build_id,
            )

    @staticmethod
    def index_collection_name(source_id: int, config: RagConfig) -> str:
        """Derive the backend-safe collection name for an index.

        Args:
            source_id:
                Canonical source identifier.
            config:
                RAG configuration whose index-affecting fields select the index.

        Returns:
            A fixed-length deterministic collection name.
        """

        identity = f"{source_id}:{index_config_fingerprint(config)}"
        return f"idx-{sha256(identity.encode('utf-8')).hexdigest()[:56]}"

    @classmethod
    def _get_rebuild_lock(cls, collection_name: str) -> Lock:
        """Return the process-local lock for one deterministic collection.

        Args:
            collection_name:
                Deterministic collection name.

        Returns:
            Shared lock serializing rebuilds of that collection.
        """

        with cls._locks_guard:
            return cls._rebuild_locks.setdefault(collection_name, Lock())

    def _create_indexing_pipeline(
        self, config: RagConfig, store: VectorStore
    ) -> IndexingPipeline:
        """Create an indexing pipeline for one source collection.

        Args:
            config:
                Persisted RAG configuration used for indexing.
            store:
                Vector store bound to the deterministic collection.

        Returns:
            Configured source-specific indexing pipeline.
        """

        return IndexingPipeline(
            splitter=create_splitter(config.splitter),
            embedder=create_embedder(config.embedder),
            store=store,
        )

    def _resolve_registered_directory(self, source: Source) -> Path:
        """Validate and return a registered source directory.

        Args:
            source:
                Persisted source metadata.

        Returns:
            Validated source directory.

        Raises:
            ValueError:
                If the persisted path no longer identifies its registered
                immediate child directory.
            FileNotFoundError:
                If the directory no longer exists.
            NotADirectoryError:
                If the path is no longer a directory.
        """

        directory = self._resolve_directory(source.name)
        if str(directory) != source.directory:
            raise ValueError(f"Source {source.id} directory is no longer valid.")
        return directory

    def _resolve_directory(self, directory_name: str) -> Path:
        """Resolve and validate one immediate corpus directory.

        Args:
            directory_name:
                Candidate directory name supplied by the client.

        Returns:
            Resolved directory path inside the configured corpus root.

        Raises:
            ValueError:
                If the value does not name an immediate child directory.
            FileNotFoundError:
                If the directory does not exist.
            NotADirectoryError:
                If the path is not a directory.
        """

        if (
            not directory_name
            or directory_name in {".", ".."}
            or os.path.sep in directory_name
        ):
            raise ValueError("Corpus directory must be an immediate child directory.")

        directory = self._corpora_dir / directory_name
        if not directory.exists():
            raise FileNotFoundError(f"Directory does not exist: {directory_name}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Expected a directory: {directory_name}")

        return directory
