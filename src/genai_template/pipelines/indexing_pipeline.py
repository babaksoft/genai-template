"""Document indexing pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from opentelemetry.instrumentation.utils import suppress_instrumentation

from genai_template.components.embeddings import (
    FastEmbedEmbeddingModel,
)
from genai_template.components.readers import TextReader
from genai_template.components.splitters import (
    DocumentSplitter,
)
from genai_template.protocols import Embedder, Splitter, VectorStore
from genai_template.schemas import IndexingResult
from genai_template.utils import Timer

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """Coordinates the document indexing workflow."""

    def __init__(
        self,
        store: VectorStore,
        reader: TextReader | None = None,
        splitter: Splitter | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        """
        Initialize the indexing pipeline.

        Args:
            store:
                Vector store.
            reader:
                Document reader.
            splitter:
                Document splitter used for chunking.
            embedder:
                Document chunk embedder.
        """

        self._reader = reader or TextReader()
        self._splitter = splitter or DocumentSplitter()
        self._embedder = embedder or FastEmbedEmbeddingModel()
        self._store = store

    def run(
        self,
        data_dir: Path,
    ) -> IndexingResult:
        """Index all supported documents in a directory.

        Args:
            data_dir:
                Directory containing the documents.

        Returns:
            Summary result from indexing.
        """

        # Indexing is intentionally outside the answer-path observability scope.
        with suppress_instrumentation(), Timer() as timer:
            documents = self._reader.load(data_dir)
            chunks = self._splitter.split(documents)
            embedded_chunks = self._embedder.embed(chunks)
            if embedded_chunks:
                self._store.upsert(embedded_chunks)
            else:
                vector_size = len(self._embedder.embed_query(""))
                self._store.create(vector_size)

        result = IndexingResult(
            documents_indexed=len(documents),
            chunks_indexed=len(chunks),
            indexing_time=timer.elapsed,
        )

        logger.info(
            "Indexed %d document(s) into %d chunk(s) in %.3f second(s).",
            result.documents_indexed,
            result.chunks_indexed,
            result.indexing_time,
        )

        return result
