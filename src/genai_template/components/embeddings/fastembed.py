"""FastEmbed embedding component."""

from __future__ import annotations

import logging

from llama_index.embeddings.fastembed import FastEmbedEmbedding

from genai_template.config import settings
from genai_template.schemas.chunk import DocumentChunk

logger = logging.getLogger(__name__)


class FastEmbedEmbeddingModel:
    """Generates embeddings using FastEmbed."""

    def __init__(self) -> None:
        """Initialize the embedding model."""

        self._embed_model = FastEmbedEmbedding(
            model_name=settings.EMBEDDING_MODEL,
        )

        logger.info(
            "Initialized FastEmbed model '%s'.",
            settings.EMBEDDING_MODEL,
        )

    def embed(
        self,
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """Generate embeddings for document chunks.

        Args:
            chunks:
                Chunks to embed.

        Returns:
            The same chunks with populated embeddings.
        """

        if not chunks:
            return []

        logger.info(
            "Generating embeddings for %d chunk(s).",
            len(chunks),
        )

        texts = [chunk.text for chunk in chunks]
        embeddings = self._embed_model.get_text_embedding_batch(
            texts=texts,
        )

        for chunk, embedding in zip(
            chunks,
            embeddings,
            strict=True,
        ):
            chunk.embedding = embedding

        logger.info(
            "Generated %d embedding(s).",
            len(chunks),
        )

        return chunks

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """Generate an embedding for a user query.

        Args:
            query:
                User query to embed.

        Returns:
            Query embedding.

        Raises:
            ValueError:
                If the query is empty or contains only whitespace.
        """

        if not query.strip():
            raise ValueError("Query must not be empty.")

        logger.info("Generating embedding for query: '%s'", query)
        embedding = self._embed_model.get_query_embedding(query)
        logger.info("Query embedding generated.")

        return embedding
