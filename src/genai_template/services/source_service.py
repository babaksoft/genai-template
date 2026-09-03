"""Corpus source lifecycle service."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from genai_template.config import settings
from genai_template.db.models import Source
from genai_template.pipelines import IndexingPipeline
from genai_template.stores.vector import ChromaStore
from genai_template.utils import utc_now


class SourceService:
    """Discover and ingest document corpora from the configured root."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        corpora_dir: Path,
    ) -> None:
        """Initialize the source service.

        Args:
            session_factory:
                Factory that creates database sessions.

            corpora_dir:
                Root directory containing one directory per corpus.
        """

        self._session_factory = session_factory
        self._corpora_dir = corpora_dir.resolve()

    def list_candidates(self) -> list[str]:
        """List immediate corpus directories available for ingestion.

        Returns:
            Sorted directory basenames under the configured corpus root.
        """

        if not self._corpora_dir.exists():
            return []

        with self._session_factory() as session:
            ingested_names = set(session.scalars(select(Source.name)))

        return sorted(
            path.name
            for path in self._corpora_dir.iterdir()
            if (
                path.is_dir()
                and path.resolve().parent == self._corpora_dir
                and path.name not in ingested_names
            )
        )

    def list_sources(self) -> list[Source]:
        """List all successfully ingested sources.

        Returns:
            Sources ordered by name.
        """

        with self._session_factory() as session:
            return list(session.scalars(select(Source).order_by(Source.name)))

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

    def _create_indexing_pipeline(self, collection_name: str) -> IndexingPipeline:
        """Create an indexing pipeline for one source collection.

        Args:
            collection_name:
                Chroma collection name for the source.

        Returns:
            Configured source-specific indexing pipeline.
        """

        return IndexingPipeline(
            store=ChromaStore(
                persist_directory=settings.CHROMA_PERSIST_DIR,
                collection_name=collection_name,
            ),
        )

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

        supplied_path = Path(directory_name)
        if supplied_path.name != directory_name or directory_name in {".", ".."}:
            raise ValueError("Corpus directory must be an immediate child directory.")

        directory = (self._corpora_dir / supplied_path).resolve()
        if directory.parent != self._corpora_dir:
            raise ValueError(
                "Corpus directory must be inside the configured corpus root."
            )
        if not directory.exists():
            raise FileNotFoundError(f"Directory does not exist: {directory_name}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Expected a directory: {directory_name}")

        return directory
