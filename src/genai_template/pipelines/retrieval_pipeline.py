"""Document retrieval pipeline."""

from __future__ import annotations

import logging

from genai_template.components.embeddings import (
    FastEmbedEmbeddingModel,
)
from genai_template.config import settings
from genai_template.observability import (
    INPUT_VALUE,
    RETRIEVAL_DOCUMENTS,
    application_span,
    retrieved_documents_attribute,
)
from genai_template.protocols import Embedder, VectorStore
from genai_template.schemas import RetrievedChunk
from genai_template.stores.vector import ChromaStore
from genai_template.utils import Timer

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Coordinates the document retrieval workflow."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        top_k: int = settings.TOP_K,
    ) -> None:
        """Initialize the retrieval pipeline.

        Args:
            embedder:
                Embedding model.
            store:
                Vector store.
            top_k:
                Default maximum number of retrieved chunks.
        """

        self._embedder = embedder or FastEmbedEmbeddingModel()
        self._store = store or ChromaStore()
        self._top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant document chunks.

        Args:
            query:
                User query.
            top_k:
                Maximum number of retrieved chunks. When omitted, use the
                value configured for this pipeline.

        Returns:
            Retrieved chunks in the order returned by the underlying
            vector store (typically increasing distance).
        """

        resolved_top_k = self._top_k if top_k is None else top_k

        with application_span(
            "rag.retrieval",
            "RETRIEVER",
            {INPUT_VALUE: query, "rag.top_k": resolved_top_k},
        ) as span:
            with Timer() as timer:
                embedding = self._embedder.embed_query(query)
                retrieved_chunks = self._store.search(
                    embedding=embedding,
                    top_k=resolved_top_k,
                    query=query,
                )
            span.set_attribute("rag.result_count", len(retrieved_chunks))
            span.set_attribute(
                RETRIEVAL_DOCUMENTS,
                retrieved_documents_attribute(
                    [
                        {
                            "id": item.chunk.id,
                            "document_id": item.chunk.document_id,
                            "content": item.chunk.text,
                            "distance": item.distance,
                        }
                        for item in retrieved_chunks
                    ]
                ),
            )

        logger.info(
            "Retrieval completed in %.3f second(s).",
            timer.elapsed,
        )

        return retrieved_chunks
