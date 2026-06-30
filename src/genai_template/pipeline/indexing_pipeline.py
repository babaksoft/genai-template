"""Document indexing pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from genai_template.components.embeddings.fastembed import (
    FastEmbedEmbeddingModel,
)
from genai_template.components.readers.text_reader import TextReader
from genai_template.components.splitters.sentence_splitter import (
    DocumentSplitter,
)
from genai_template.schemas.chunk import DocumentChunk
from genai_template.stores.vector.chroma_store import ChromaStore

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
        """Initialize the indexing pipeline."""

        self._reader = reader or TextReader()
        self._splitter = splitter or DocumentSplitter()
        self._embedder = embedder or FastEmbedEmbeddingModel()
        self._store = store or ChromaStore()

    def run(
        self,
        data_dir: Path,
    ) -> list[DocumentChunk]:
        """Index all supported documents in a directory.

        Args:
            data_dir:
                Directory containing the documents.

        Returns:
            Indexed document chunks.
        """

        logger.info(
            "Starting indexing pipeline for '%s'.",
            data_dir,
        )

        documents = self._reader.load(data_dir)
        chunks = self._splitter.split(documents)
        embedded_chunks = self._embedder.embed(chunks)
        self._store.upsert(embedded_chunks)

        logger.info(
            "Indexed %d chunk(s).",
            len(embedded_chunks),
        )

        return embedded_chunks
