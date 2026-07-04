"""Document retrieval pipeline."""

from __future__ import annotations

import logging

from genai_template.components.embeddings.fastembed import (
    FastEmbedEmbeddingModel,
)
from genai_template.schemas.retrieved_chunk import RetrievedChunk
from genai_template.stores.vector.chroma_store import ChromaStore

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
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant document chunks.

        Args:
            query:
                User query.
            top_k:
                Maximum number of retrieved chunks. Default is 5.

        Returns:
            Returns retrieved chunks in the order returned by the underlying
            vector store (typically increasing distance).
        """

        logger.info("Retrieving relevant document chunks.")

        embedding = self._embedder.embed_query(query)

        retrieved_chunks = self._store.search(
            embedding=embedding,
            top_k=top_k,
        )

        logger.info(
            "Retrieved %d chunk(s).",
            len(retrieved_chunks),
        )

        return retrieved_chunks
