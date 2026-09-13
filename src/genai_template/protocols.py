"""Structural contracts for factory-created RAG components."""

from typing import Protocol

from llama_index.core import Document

from genai_template.schemas import DocumentChunk, RetrievedChunk


class Splitter(Protocol):
    """Contract for splitting documents into canonical chunks."""

    def split(self, documents: list[Document]) -> list[DocumentChunk]:
        """Split documents into chunks.

        Args:
            documents:
                Documents to split.

        Returns:
            Document chunks produced from the supplied documents.
        """

        ...


class Embedder(Protocol):
    """Contract for embedding document chunks and queries."""

    def embed(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Generate embeddings for document chunks.

        Args:
            chunks:
                Chunks to embed.

        Returns:
            Document chunks with populated embeddings.
        """

        ...

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a query.

        Args:
            query:
                Query to embed.

        Returns:
            Query embedding.
        """

        ...


class VectorStore(Protocol):
    """Contract for vector-store lifecycle and retrieval operations."""

    def create(self, vector_size: int) -> None:
        """Create an empty collection when one does not already exist.

        Args:
            vector_size:
                Dimensionality of vectors that the collection will store.
        """

        ...

    def exists(self) -> bool:
        """Return whether the vector store's collection exists.

        Returns:
            ``True`` when the collection exists, including when it is empty.
        """

        ...

    def count(self) -> int:
        """Return the number of stored vector records.

        Returns:
            Number of stored vector records.
        """

        ...

    def delete(self) -> None:
        """Delete the vector store's collection."""

        ...

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        """Persist or update embedded document chunks.

        Args:
            chunks:
                Embedded chunks to persist or update.
        """

        ...

    def search(
        self,
        embedding: list[float],
        top_k: int,
        query: str | None = None,
    ) -> list[RetrievedChunk]:
        """Search for similar document chunks.

        Args:
            embedding:
                Query embedding.
            top_k:
                Maximum number of results.
            query:
                Optional query string used for observability.

        Returns:
            Retrieved chunks ordered by the store's relevance measure.
        """

        ...


class LanguageModel(Protocol):
    """Contract for generating text from a prompt."""

    def generate(self, prompt: str) -> str:
        """Generate a response for a prompt.

        Args:
            prompt:
                Prompt to send to the language model.

        Returns:
            Generated response.
        """

        ...


class Retriever(Protocol):
    """Contract for retrieving document chunks for a query."""

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant document chunks.

        Args:
            query:
                Query used for retrieval.
            top_k:
                Optional maximum number of chunks to retrieve.

        Returns:
            Retrieved chunks in the implementation-defined relevance order.
        """

        ...


__all__ = [
    "Embedder",
    "LanguageModel",
    "Retriever",
    "Splitter",
    "VectorStore",
]
