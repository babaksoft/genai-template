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
from genai_template.schemas import RetrievedChunk
from genai_template.stores.vector import ChromaStore
from genai_template.utils import Timer

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Coordinates the document retrieval workflow."""

    def __init__(
        self,
        embedder: FastEmbedEmbeddingModel | None = None,
        store: ChromaStore | None = None,
    ) -> None:
        """Initialize the retrieval pipeline.

        Args:
            embedder:
                Embedding model.
            store:
                Vector store.
        """

        self._embedder = embedder or FastEmbedEmbeddingModel()
        self._store = store or ChromaStore()

    def retrieve(
        self,
        query: str,
        top_k: int = settings.TOP_K,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant document chunks.

        Args:
            query:
                User query.
            top_k:
                Maximum number of retrieved chunks. Default is 5.

        Returns:
            Retrieved chunks in the order returned by the underlying
            vector store (typically increasing distance).
        """

        with application_span(
            "rag.retrieval",
            "RETRIEVER",
            {INPUT_VALUE: query, "rag.top_k": top_k},
        ) as span:
            with Timer() as timer:
                embedding = self._embedder.embed_query(query)
                retrieved_chunks = self._store.search(
                    embedding=embedding,
                    top_k=top_k,
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
