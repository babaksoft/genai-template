"""Persistent Chroma vector store."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import cast

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Embeddings, Metadatas

from genai_template.common.types import VectorDistance
from genai_template.config import settings
from genai_template.observability import (
    INPUT_VALUE,
    RETRIEVAL_DOCUMENTS,
    application_span,
    retrieved_documents_attribute,
)
from genai_template.schemas import DocumentChunk, RetrievedChunk
from genai_template.utils import Timer

logger = logging.getLogger(__name__)


class ChromaStore:
    """Persistent Chroma vector store."""

    def __init__(
        self,
        persist_directory: Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        """
        Initialize the Chroma collection.

        Args:
            persist_directory:
                Directory to store persisted data.
            collection_name:
                Name of the collection to use. Defaults to the configured
                application collection.
        """

        self._DISTANCE_MAP = {
            VectorDistance.COSINE: "cosine",
            VectorDistance.L2: "l2",
            VectorDistance.INNER_PRODUCT: "ip",
        }

        persist_directory = persist_directory or settings.CHROMA_PERSIST_DIR
        collection_name = collection_name or settings.CHROMA_COLLECTION

        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection_name = collection_name
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": self._DISTANCE_MAP[settings.CHROMA_DISTANCE],
            },
        )

        logger.info(
            "Connected to Chroma collection '%s'.",
            collection_name,
        )

    def delete(self) -> None:
        """Delete this store's Chroma collection."""

        self._client.delete_collection(name=self._collection_name)

    def upsert(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        """Persist or update (upsert) embedded document chunks.

        Args:
            chunks:
                Embedded chunks to persist or update.

        Raises:
            ValueError:
                If a chunk has no embedding.
        """

        if not chunks:
            return

        ids: list[str] = []
        documents: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, str]] = []

        with Timer() as timer:
            for chunk in chunks:
                if chunk.embedding is None:
                    raise ValueError(f"Chunk '{chunk.id}' has no embedding.")

                ids.append(chunk.id)
                documents.append(chunk.text)
                embeddings.append(chunk.embedding)

                metadata = {
                    "document_id": chunk.document_id,
                    "metadata": json.dumps(chunk.metadata),
                }

                metadatas.append(metadata)

            self._collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=cast(Embeddings, embeddings),
                metadatas=cast(Metadatas, metadatas),
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

        Args:
            embedding:
                Query embedding.
            top_k:
                Maximum number of results.

        Returns:
            Retrieved chunks ordered by increasing distance.
        """

        logger.info("Searching vector store for similar chunks: top_k=%d", top_k)

        with (
            application_span(
                "rag.chroma.search",
                "RETRIEVER",
                {INPUT_VALUE: query, "rag.top_k": top_k},
            ) as span,
            Timer() as timer,
        ):
            result = self._collection.query(
                query_embeddings=cast(Embeddings, [embedding]),
                n_results=top_k,
            )

            ids = result.get("ids") or []
            documents = result.get("documents") or []
            metadatas = result.get("metadatas") or []
            distances = result.get("distances") or []

            retrieved_chunks: list[RetrievedChunk] = []
            if ids:
                for chunk_id, text, metadata, distance in zip(
                    ids[0],
                    documents[0],
                    metadatas[0],
                    distances[0],
                    strict=True,
                ):
                    document_chunk = DocumentChunk(
                        id=chunk_id,
                        document_id=str(metadata["document_id"]),
                        text=text,
                        metadata=json.loads(str(metadata["metadata"])),
                    )

                    retrieved_chunks.append(
                        RetrievedChunk(
                            chunk=document_chunk,
                            distance=distance,
                        )
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

        self._log_stats(retrieved_chunks, timer)

        return retrieved_chunks

    def _log_stats(self, retrieved_chunks: list[RetrievedChunk], timer: Timer) -> None:

        logger.info(
            "Retrieved %d chunk(s) in %.3f second(s).",
            len(retrieved_chunks),
            timer.elapsed,
        )

        if retrieved_chunks:
            distances = [chunk.distance for chunk in retrieved_chunks]

            distances.sort()
            logger.info("Distance range: [%.2f, %.2f]", distances[0], distances[-1])
