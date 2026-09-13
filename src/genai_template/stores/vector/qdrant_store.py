"""Qdrant-backed vector store."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, cast
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from genai_template.common.types import VectorDistance
from genai_template.observability import (
    INPUT_VALUE,
    RETRIEVAL_DOCUMENTS,
    application_span,
    retrieved_documents_attribute,
)
from genai_template.schemas import DocumentChunk, RetrievedChunk
from genai_template.utils import Timer

logger = logging.getLogger(__name__)


class QdrantStore:
    """Vector store supporting local and remote Qdrant deployments."""

    _DISTANCE_MAP: ClassVar[dict[VectorDistance, models.Distance]] = {
        VectorDistance.COSINE: models.Distance.COSINE,
        VectorDistance.INNER_PRODUCT: models.Distance.DOT,
        VectorDistance.L2: models.Distance.EUCLID,
    }

    def __init__(
        self,
        collection_name: str,
        distance: VectorDistance,
        *,
        path: Path | None = None,
        url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize a local or server-backed Qdrant store.

        Exactly one of ``path`` and ``url`` must be supplied. Collections are
        intentionally not created until the first non-empty upsert, when the
        embedding dimensionality is known.

        Args:
            collection_name:
                Name of the Qdrant collection.
            distance:
                Distance metric used by the collection.
            path:
                File system path for local Qdrant persistence.
            url:
                HTTP(S) URL for a Qdrant server.
            api_key:
                Optional API key for a Qdrant server.

        Raises:
            ValueError:
                If both or neither connection modes are supplied.
        """

        if (path is None) == (url is None):
            raise ValueError("Exactly one of path or url must be provided.")
        if path is not None and api_key is not None:
            raise ValueError("api_key is only supported for server Qdrant.")

        if path is not None:
            self._client = QdrantClient(path=str(path))
        else:
            self._client = QdrantClient(url=url, api_key=api_key)

        self._collection_name = collection_name
        self._distance = distance

        logger.info("Connected to Qdrant collection '%s'.", collection_name)

    def delete(self) -> None:
        """Delete this store's collection if it exists."""

        if self.exists():
            self._client.delete_collection(collection_name=self._collection_name)

    def create(self, vector_size: int) -> None:
        """Create an empty collection if it does not already exist.

        Args:
            vector_size:
                Dimensionality of vectors stored in the collection.

        Raises:
            ValueError:
                If ``vector_size`` is not positive.
        """

        if vector_size <= 0:
            raise ValueError("vector_size must be positive")
        if not self.exists():
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=self._DISTANCE_MAP[self._distance],
                ),
            )

    def exists(self) -> bool:
        """Return whether the configured Qdrant collection exists.

        Returns:
            ``True`` when the collection exists, including when it is empty.
        """

        return self._client.collection_exists(collection_name=self._collection_name)

    def count(self) -> int:
        """Return the number of records in this store's collection.

        Returns:
            Number of stored vector records, or zero before collection creation.
        """

        if not self.exists():
            return 0

        result = self._client.count(
            collection_name=self._collection_name,
            exact=True,
        )

        return result.count

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        """Persist or update embedded document chunks.

        All chunks are validated before the collection is created or any
        points are written.

        Args:
            chunks:
                Embedded chunks to persist or update.

        Raises:
            ValueError:
                If an embedding is absent, empty, or has a different dimension.
        """

        if not chunks:
            return

        vector_size = self._validate_embeddings(chunks)
        points = [
            models.PointStruct(
                id=self._point_id(chunk.id),
                vector=cast(list[float], chunk.embedding),
                payload={
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                },
            )
            for chunk in chunks
        ]

        with Timer() as timer:
            self.create(vector_size)
            self._client.upsert(
                collection_name=self._collection_name,
                points=points,
            )

        logger.info(
            "Stored %d chunk(s) in %.3f second(s).",
            len(chunks),
            timer.elapsed,
        )

    def search(
        self,
        embedding: list[float],
        top_k: int,
        query: str | None = None,
    ) -> list[RetrievedChunk]:
        """Search for similar document chunks.

        Qdrant similarity scores are normalized to the project convention in
        which smaller distances represent closer results.

        Args:
            embedding:
                Query embedding.
            top_k:
                Maximum number of results.
            query:
                Optional query string for observability span attributes.

        Returns:
            Retrieved chunks ordered by increasing normalized distance.
        """

        logger.info("Searching vector store for similar chunks: top_k=%d", top_k)

        with (
            application_span(
                "rag.qdrant.search",
                "RETRIEVER",
                {INPUT_VALUE: query, "rag.top_k": top_k},
            ) as span,
            Timer() as timer,
        ):
            retrieved_chunks: list[RetrievedChunk] = []
            if self.exists():
                response: Any = self._client.query_points(
                    collection_name=self._collection_name,
                    query=embedding,
                    limit=top_k,
                    with_payload=True,
                )
                retrieved_chunks = [
                    self._retrieved_chunk(point.payload, point.score)
                    for point in response.points
                ]
                retrieved_chunks.sort(key=lambda item: item.distance)

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

        self._log_stats(retrieved_chunks, timer)
        return retrieved_chunks

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        """Convert a canonical chunk identifier to a stable Qdrant UUID.

        Args:
            chunk_id:
                Canonical chunk identifier.

        Returns:
            Deterministic UUID string accepted by Qdrant.
        """

        return str(uuid5(NAMESPACE_URL, chunk_id))

    @staticmethod
    def _validate_embeddings(chunks: list[DocumentChunk]) -> int:
        """Validate every embedding and return their common dimension.

        Args:
            chunks:
                Chunks to validate.

        Returns:
            Common embedding dimension.

        Raises:
            ValueError:
                If an embedding is absent, empty, or has a different dimension.
        """

        vector_size: int | None = None
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk '{chunk.id}' has no embedding.")
            if not chunk.embedding:
                raise ValueError(f"Chunk '{chunk.id}' has an empty embedding.")
            if vector_size is None:
                vector_size = len(chunk.embedding)
            elif len(chunk.embedding) != vector_size:
                raise ValueError("All chunk embeddings must have the same dimension.")

        if vector_size is None:  # pragma: no cover - guarded by the caller
            raise ValueError("At least one chunk is required.")
        return vector_size

    def _retrieved_chunk(
        self,
        payload: dict[str, Any] | None,
        score: float,
    ) -> RetrievedChunk:
        """Convert a Qdrant scored point into the canonical result schema.

        Args:
            payload:
                Stored Qdrant point payload.
            score:
                Provider-native result score.

        Returns:
            Canonical retrieved chunk with normalized distance.

        Raises:
            ValueError:
                If the point has no payload.
        """

        if payload is None:
            raise ValueError("Qdrant result has no payload.")

        chunk = DocumentChunk(
            id=str(payload["chunk_id"]),
            document_id=str(payload["document_id"]),
            text=str(payload["text"]),
            metadata=payload.get("metadata") or {},
        )
        return RetrievedChunk(
            chunk=chunk,
            distance=self._normalized_distance(score),
        )

    def _normalized_distance(self, score: float) -> float:
        """Normalize a Qdrant score to the project's distance convention.

        Args:
            score:
                Qdrant result score.

        Returns:
            Distance where a smaller value represents a closer match.
        """

        if self._distance is VectorDistance.COSINE:
            return 1.0 - score
        if self._distance is VectorDistance.INNER_PRODUCT:
            return -score
        return score

    @staticmethod
    def _log_stats(retrieved_chunks: list[RetrievedChunk], timer: Timer) -> None:
        """Log retrieval duration and distance range.

        Args:
            retrieved_chunks:
                Retrieved results.
            timer:
                Completed retrieval timer.
        """

        logger.info(
            "Retrieved %d chunk(s) in %.3f second(s).",
            len(retrieved_chunks),
            timer.elapsed,
        )
        if retrieved_chunks:
            distances = sorted(chunk.distance for chunk in retrieved_chunks)
            logger.info("Distance range: [%.2f, %.2f]", distances[0], distances[-1])
