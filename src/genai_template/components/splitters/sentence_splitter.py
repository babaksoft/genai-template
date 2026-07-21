"""Document splitter built on top of LlamaIndex's SentenceSplitter."""

from __future__ import annotations

import logging

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from genai_template.config import settings
from genai_template.schemas import DocumentChunk
from genai_template.utils import Timer

logger = logging.getLogger(__name__)


class DocumentSplitter:
    """Splits documents into canonical document chunks."""

    def __init__(self) -> None:
        """Initialize the document splitter."""

        self._splitter = SentenceSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

    def split(
        self,
        documents: list[Document],
    ) -> list[DocumentChunk]:
        """Split documents into chunks.

        Args:
            documents:
                Documents to split.

        Returns:
            A list of document chunks.
        """

        if not documents:
            return []

        logger.info("Splitting %d document(s).", len(documents))

        with Timer() as timer:
            nodes = self._splitter.get_nodes_from_documents(documents)

            chunk_counts: dict[str, int] = {}
            chunks: list[DocumentChunk] = []

            for node in nodes:
                metadata = dict(node.metadata)

                document_id = str(
                    metadata.get(
                        "file_name",
                        metadata.get("doc_id", "document"),
                    )
                )

                index = chunk_counts.get(document_id, 0)
                chunk_counts[document_id] = index + 1

                chunk = DocumentChunk(
                    id=f"{document_id}-{index:03d}",
                    document_id=document_id,
                    text=node.get_content(),
                    metadata=metadata,
                )

                chunks.append(chunk)

        logger.info(
            "Generated %d chunk(s) in %.3f second(s).",
            len(chunks),
            timer.elapsed,
        )
        self._log_stats(chunks)

        return chunks

    def _log_stats(self, chunks: list[DocumentChunk]) -> None:
        """
        Log basic statistics for given chunks.

        Args:
            chunks:
                Document chunks for logging statistics.
        """

        counts = [len(chunk.text) for chunk in chunks]
        counts.sort()

        logger.info(
            "Chunk length statistics (characters): minimum=%d maximum=%d average=%d",
            counts[0],
            counts[-1],
            round(sum(counts) / len(counts)),
        )
