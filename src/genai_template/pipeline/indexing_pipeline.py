"""Document indexing pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from genai_template.components.embeddings import (
    FastEmbedEmbeddingModel,
)
from genai_template.components.readers import TextReader
from genai_template.components.splitters import (
    DocumentSplitter,
)
from genai_template.schemas import IndexingResult
from genai_template.stores.vector import ChromaStore
from genai_template.utils import Timer

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """Coordinates the document indexing workflow."""

    def __init__(
        self,
        reader: TextReader | None = None,
        splitter: DocumentSplitter | None = None,
        embedder: FastEmbedEmbeddingModel | None = None,
        store: ChromaStore | None = None,
    ) -> None:
        """
        Initialize the indexing pipeline.

        Args:
            reader:
                Document reader.
            splitter:
                Document splitter used for chunking.
            embedder:
                Document chunk embedder.
            store:
                Vector store.
        """

        self._reader = reader or TextReader()
        self._splitter = splitter or DocumentSplitter()
        self._embedder = embedder or FastEmbedEmbeddingModel()
        self._store = store or ChromaStore()

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

        with Timer() as timer:
            documents = self._reader.load(data_dir)
            chunks = self._splitter.split(documents)
            embedded_chunks = self._embedder.embed(chunks)
            self._store.upsert(embedded_chunks)

        logger.info(
            "Indexed %d document(s) into %d chunk(s) in %.3f second(s).",
            len(documents),
            len(chunks),
            timer.elapsed,
        )

        return IndexingResult(
            documents_indexed=len(documents),
            chunks_indexed=len(chunks),
            indexing_time=timer.elapsed,
        )
