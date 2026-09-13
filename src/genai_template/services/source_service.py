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

from genai_template.config import RagConfig, index_config_fingerprint
from genai_template.db.models import Source
from genai_template.factories.embedder_factory import create_embedder
from genai_template.factories.splitter_factory import create_splitter
from genai_template.factories.vector_store_factory import create_vector_store
from genai_template.pipelines import IndexingPipeline
from genai_template.protocols import VectorStore
from genai_template.schemas import IndexingResult
from genai_template.services.rag_config_service import RagConfigService

logger = logging.getLogger(__name__)


class SourceService:
    """Register document corpora and manage their deterministic indexes."""

    _locks_guard: ClassVar[Lock] = Lock()
    _rebuild_locks: ClassVar[dict[str, Lock]] = {}

    def __init__(
        self,
        session_factory: Callable[[], Session],
        corpora_dir: Path,
        rag_config_service: RagConfigService | None = None,
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
        """

        self._session_factory = session_factory
        self._corpora_dir = corpora_dir.resolve()
        self._rag_config_service = rag_config_service or RagConfigService(
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
        """

        source = self.get_source(source_id)
        directory = self._resolve_registered_directory(source)
        record = self._rag_config_service.get_config(rag_config_id)
        config = self._rag_config_service.parse_config(record)
        collection_name = self.index_collection_name(source_id, config)
        rebuild_lock = self._get_rebuild_lock(collection_name)

        with rebuild_lock:
            store = create_vector_store(config.vector_store, collection_name)
            store.delete()
            pipeline = self._create_indexing_pipeline(config, store)
            result = pipeline.run(directory)

        logger.info(
            "Rebuilt source %d index '%s' with RAG config %d.",
            source_id,
            collection_name,
            rag_config_id,
        )
        return result

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
