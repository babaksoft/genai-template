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
from genai_template.schemas.chunk import DocumentChunk
from genai_template.schemas.retrieved_chunk import RetrievedChunk

logger = logging.getLogger(__name__)


class ChromaStore:
    """Persistent Chroma vector store."""

    _DISTANCE_MAP = {
        VectorDistance.COSINE: "cosine",
        VectorDistance.L2: "l2",
        VectorDistance.INNER_PRODUCT: "ip",
    }

    def __init__(self, persist_directory: Path | None = None) -> None:
        """
        Initialize the Chroma collection.

        Args:
            persist_directory:
                Directory to store persisted data.
        """

        persist_directory = persist_directory or settings.CHROMA_PERSIST_DIR
        client = chromadb.PersistentClient(path=persist_directory)

        self._collection: Collection = client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={
                "hnsw:space": self._DISTANCE_MAP[settings.CHROMA_DISTANCE],
            },
        )

        logger.info(
            "Connected to Chroma collection '%s'.",
            settings.CHROMA_COLLECTION,
        )

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
            "Stored %d chunk(s).",
            len(chunks),
        )

    def search(
        self,
        embedding: list[float],
        top_k: int,
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

        result = self._collection.query(
            query_embeddings=cast(Embeddings, [embedding]),
            n_results=top_k,
        )

        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        distances = result.get("distances") or []

        if not ids:
            return []

        retrieved_chunks: list[RetrievedChunk] = []

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

        logger.info(
            "Retrieved %d chunk(s).",
            len(retrieved_chunks),
        )

        return retrieved_chunks
