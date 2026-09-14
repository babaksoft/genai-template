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
from genai_template.observability import application_span
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

        with application_span("rag.index.build", "CHAIN") as build_span:
            with Timer() as timer:
                with application_span("rag.index.load", "CHAIN") as load_span:
                    documents = self._reader.load(data_dir)
                    load_span.set_attribute("rag.document.count", len(documents))

                with application_span(
                    "rag.index.split",
                    "CHAIN",
                    {"rag.document.count": len(documents)},
                ) as split_span:
                    chunks = self._splitter.split(documents)
                    split_span.set_attribute("rag.chunk.count", len(chunks))

                with application_span(
                    "rag.index.embed",
                    "EMBEDDING",
                    {"rag.chunk.count": len(chunks)},
                ) as embed_span:
                    embedded_chunks = self._embedder.embed(chunks)
                    embed_span.set_attribute(
                        "rag.embedding.count", len(embedded_chunks)
                    )
                    if embedded_chunks and embedded_chunks[0].embedding is not None:
                        embed_span.set_attribute(
                            "rag.embedding.dimension",
                            len(embedded_chunks[0].embedding),
                        )

                write_operation = "upsert" if embedded_chunks else "create"
                with application_span(
                    "rag.index.write",
                    "CHAIN",
                    {
                        "rag.index.write.operation": write_operation,
                    },
                ) as write_span:
                    if embedded_chunks:
                        self._store.upsert(embedded_chunks)
                    else:
                        vector_size = len(self._embedder.embed_query(""))
                        self._store.create(vector_size)
                        write_span.set_attribute("rag.embedding.dimension", vector_size)
                    write_span.set_attribute(
                        "rag.index.write.count", len(embedded_chunks)
                    )

            build_span.set_attribute("rag.document.count", len(documents))
            build_span.set_attribute("rag.chunk.count", len(chunks))

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
