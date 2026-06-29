"""Persistent Chroma vector store."""

from __future__ import annotations

import json
import logging
from typing import cast

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Embeddings, Metadatas

from genai_template.common.types import VectorDistance
from genai_template.config import settings
from genai_template.schemas.chunk import DocumentChunk

logger = logging.getLogger(__name__)


class ChromaStore:
    """Persistent Chroma vector store."""

    _DISTANCE_MAP = {
        VectorDistance.COSINE: "cosine",
        VectorDistance.L2: "l2",
        VectorDistance.INNER_PRODUCT: "ip",
    }

    def __init__(self) -> None:
        """Initialize the Chroma collection."""
        client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
        )

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

    def add(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        """Persist embedded document chunks.

        Args:
            chunks:
                Embedded chunks to persist.

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

        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=cast(Embeddings, embeddings),
            metadatas=cast(Metadatas, metadatas),
        )

        logger.info(
            "Stored %d chunk(s).",
            len(chunks),
        )

    def query(
        self,
        embedding: list[float],
        top_k: int,
    ) -> list[str]:
        """Query the vector store.

        Args:
            embedding:
                Query embedding.
            top_k:
                Maximum number of matches.

        Returns:
            Matching chunk IDs.
        """
        result = self._collection.query(
            query_embeddings=cast(Embeddings, [embedding]),
            n_results=top_k,
        )

        ids = result.get("ids", [])

        if not ids:
            return []

        return list(ids[0])
