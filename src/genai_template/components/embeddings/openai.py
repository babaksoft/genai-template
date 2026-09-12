"""OpenAI embedding component."""

from __future__ import annotations

import logging

from llama_index.embeddings.openai import OpenAIEmbedding

from genai_template.config import settings
from genai_template.schemas import DocumentChunk
from genai_template.utils import Timer

logger = logging.getLogger(__name__)


class OpenAIEmbeddingModel:
    """Generate document and query embeddings with OpenAI."""

    def __init__(
        self,
        model_name: str,
        dimensions: int | None = None,
        request_timeout: float = settings.REQUEST_TIMEOUT,
    ) -> None:
        """Initialize the OpenAI embedding model.

        Args:
            model_name:
                Name of the OpenAI embedding model.
            dimensions:
                Optional number of dimensions in each returned embedding.
            request_timeout:
                Maximum number of seconds to wait for an OpenAI request.
        """

        self._embed_model = OpenAIEmbedding(
            model=model_name,
            dimensions=dimensions,
            timeout=request_timeout,
        )

        logger.info("Initialized OpenAI embedding model '%s'.", model_name)

    def embed(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Generate embeddings for document chunks.

        Args:
            chunks:
                Chunks to embed.

        Returns:
            The same chunks with populated embeddings.
        """

        if not chunks:
            return []

        logger.info("Generating embeddings for %d chunk(s).", len(chunks))

        with Timer() as timer:
            embeddings = self._embed_model.get_text_embedding_batch(
                texts=[chunk.text for chunk in chunks],
            )

            for chunk, embedding in zip(chunks, embeddings, strict=True):
                chunk.embedding = embedding

        logger.info(
            "Generated %d embedding(s) in %.3f second(s): dimension=%d",
            len(chunks),
            timer.elapsed,
            len(chunks[0].embedding or []),
        )

        return chunks

    def embed_query(self, query: str) -> list[float]:
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
        logger.info("Query length: %d character(s)", len(query))

        with Timer() as timer:
            embedding = self._embed_model.get_query_embedding(query)

        logger.info("Query embedding generated in %.3f second(s).", timer.elapsed)

        return embedding
