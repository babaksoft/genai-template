"""Corpus source lifecycle service."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from genai_template.config import RagConfig, load_rag_config
from genai_template.db.models import Source
from genai_template.factories.embedder_factory import create_embedder
from genai_template.factories.splitter_factory import create_splitter
from genai_template.factories.vector_store_factory import create_vector_store
from genai_template.pipelines import IndexingPipeline
from genai_template.utils import utc_now

logger = logging.getLogger(__name__)


class SourceService:
    """Discover and ingest document corpora from the configured root."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        corpora_dir: Path,
        config: RagConfig | None = None,
    ) -> None:
        """Initialize the source service.

        Args:
            session_factory:
                Factory that creates database sessions.
            corpora_dir:
                Root directory containing one directory per corpus.
            config:
                Resolved RAG configuration used to construct source indexes.
                Application defaults are loaded when omitted.
        """

        self._session_factory = session_factory
        self._corpora_dir = corpora_dir.resolve()
        self._config = config or load_rag_config()

    def list_candidates(self) -> list[str]:
        """List immediate corpus directories available for ingestion.

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
        """List all successfully ingested sources.

        Returns:
            Sources ordered by name.
        """

        with self._session_factory() as session:
            return list(session.scalars(select(Source).order_by(Source.name)))

    def get_source(self, source_id: int) -> Source:
        """Get an ingested source by identifier.

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

    def ingest(self, directory_name: str) -> Source:
        """Ingest one previously prepared corpus directory.

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

        collection_name = f"source-{uuid4().hex}"
        indexing_pipeline = self._create_indexing_pipeline(collection_name)
        result = indexing_pipeline.run(directory)

        source = Source(
            name=source_name,
            directory=str(directory),
            collection_name=collection_name,
            documents_indexed=result.documents_indexed,
            chunks_indexed=result.chunks_indexed,
            indexed_at=utc_now(),
            indexing_time=result.indexing_time,
        )

        try:
            with self._session_factory() as session:
                session.add(source)
                session.commit()
                session.refresh(source)
        except IntegrityError as exc:
            raise ValueError(f"Source '{source_name}' already exists.") from exc

        return source

    def refresh(self, source_id: int) -> Source:
        """Rebuild one source in a new vector collection.

        Args:
            source_id:
                Identifier of the source to rebuild.

        Returns:
            Refreshed source metadata.

        Raises:
            ValueError:
                If no source has the supplied identifier or
                if the source directory is no longer valid.
            FileNotFoundError:
                If the source directory no longer exists.
            NotADirectoryError:
                If the source path is no longer a directory.
        """

        source = self.get_source(source_id)
        directory = self._resolve_directory(source.name)
        new_collection_name = f"source-{uuid4().hex}"

        try:
            indexing_pipeline = self._create_indexing_pipeline(new_collection_name)
            result = indexing_pipeline.run(directory)
        except Exception:
            try:
                self._delete_collection(new_collection_name)
            except Exception:
                logger.exception(
                    "Could not rollback new collection '%s'.",
                    new_collection_name,
                )
            raise

        old_collection_name = source.collection_name
        with self._session_factory() as session:
            persisted_source = session.get(Source, source_id)
            if persisted_source is None:
                raise ValueError(f"Source {source_id} does not exist.")

            persisted_source.collection_name = new_collection_name
            persisted_source.documents_indexed = result.documents_indexed
            persisted_source.chunks_indexed = result.chunks_indexed
            persisted_source.indexed_at = utc_now()
            persisted_source.indexing_time = result.indexing_time
            session.commit()
            session.refresh(persisted_source)

        try:
            self._delete_collection(old_collection_name)
        except Exception:
            logger.exception(
                "Refreshed source %d but could not delete old collection '%s'.",
                source_id,
                old_collection_name,
            )

        return persisted_source

    def _create_indexing_pipeline(self, collection_name: str) -> IndexingPipeline:
        """Create an indexing pipeline for one source collection.

        Args:
            collection_name:
                Chroma collection name for the source.

        Returns:
            Configured source-specific indexing pipeline.
        """

        return IndexingPipeline(
            splitter=create_splitter(self._config.splitter),
            embedder=create_embedder(self._config.embedder),
            store=create_vector_store(
                self._config.vector_store,
                collection_name,
            ),
        )

    def _delete_collection(self, collection_name: str) -> None:
        """Delete a source-specific Chroma collection.

        Args:
            collection_name:
                Name of the collection to delete.
        """

        create_vector_store(
            self._config.vector_store,
            collection_name,
        ).delete()

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
