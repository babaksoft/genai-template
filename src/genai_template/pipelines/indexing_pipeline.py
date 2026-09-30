"""Document indexing pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from genai_template.components.embeddings import (
    FastEmbedEmbeddingModel,
)
from genai_template.components.readers import PortfolioReader, TextReader
from genai_template.components.readers.portfolio_metadata import (
    CORPUS_FINGERPRINT,
    validate_portfolio_metadata,
)
from genai_template.components.splitters import (
    DocumentSplitter,
)
from genai_template.components.splitters.identity import validate_unique_chunk_ids
from genai_template.observability import application_span
from genai_template.protocols import Embedder, Reader, Splitter, VectorStore
from genai_template.schemas import DocumentChunk, IndexingResult, LoadedDocuments
from genai_template.utils import Timer

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """Coordinates the document indexing workflow."""

    def __init__(
        self,
        store: VectorStore,
        reader: Reader | None = None,
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

        self._reader = reader
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
                    loaded = self._load(data_dir)
                    documents = list(loaded.documents)
                    load_span.set_attribute("rag.document.count", len(documents))

                with application_span(
                    "rag.index.split",
                    "CHAIN",
                    {"rag.document.count": len(documents)},
                ) as split_span:
                    chunks = self._splitter.split(documents)
                    validate_unique_chunk_ids(chunks)
                    self._validate_portfolio_chunks(loaded, chunks)
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
            corpus_fingerprint=self._corpus_fingerprint(loaded),
        )

        logger.info(
            "Indexed %d document(s) into %d chunk(s) in %.3f second(s).",
            result.documents_indexed,
            result.chunks_indexed,
            result.indexing_time,
        )

        return result

    def _load(self, data_dir: Path) -> LoadedDocuments:
        """Load through an explicit or automatically selected reader.

        Args:
            data_dir:
                Source directory or Portfolio publication pointer.

        Returns:
            Typed reader result.
        """

        reader = self._reader
        if reader is None:
            reader = (
                PortfolioReader()
                if (data_dir / "manifest.json").is_file()
                else TextReader()
            )
        loaded = reader.load(data_dir)
        if isinstance(loaded, LoadedDocuments):
            return loaded

        # Preserve compatibility with test doubles and older reader adapters.
        return LoadedDocuments(documents=tuple(loaded))

    @staticmethod
    def _corpus_fingerprint(loaded: LoadedDocuments) -> str | None:
        """Extract the fingerprint exposed by normalized provenance.

        Args:
            loaded:
                Typed reader result.

        Returns:
            Corpus fingerprint when provenance supplies one.
        """

        if loaded.provenance is None:
            return None

        return loaded.provenance.corpus_fingerprint

    @staticmethod
    def _validate_portfolio_chunks(
        loaded: LoadedDocuments,
        chunks: list[DocumentChunk],
    ) -> None:
        """Validate provenance and deterministic identities before embedding.

        Args:
            loaded:
                Reader result that identifies manifest-backed input.
            chunks:
                Splitter output to validate.

        Raises:
            ValueError:
                If a Portfolio chunk loses provenance or has an unstable identity.
        """

        provenance = loaded.provenance
        if provenance is None:
            return

        document_ids = {document.id_ for document in loaded.documents}
        chunk_counts: dict[str, int] = {}
        for chunk in chunks:
            validate_portfolio_metadata(chunk.metadata, allow_header_path=True)
            if chunk.document_id not in document_ids:
                raise ValueError("Portfolio chunk references an unknown document")
            if chunk.metadata[CORPUS_FINGERPRINT] != provenance.corpus_fingerprint:
                raise ValueError("Portfolio chunk has the wrong corpus fingerprint")
            ordinal = chunk_counts.get(chunk.document_id, 0)
            expected_id = f"{chunk.document_id}-{ordinal:03d}"
            if chunk.id != expected_id:
                raise ValueError("Portfolio chunk ID is not deterministic")
            chunk_counts[chunk.document_id] = ordinal + 1
