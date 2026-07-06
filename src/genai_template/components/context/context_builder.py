"""Context builder component."""

import logging

from genai_template.schemas.retrieved_chunk import RetrievedChunk
from genai_template.utils.timer import Timer

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Build an LLM-ready context string from retrieved chunks."""

    def build(self, retrieved_chunks: list[RetrievedChunk]) -> str:
        """
        Build a context string from retrieved chunks.

        Args:
            retrieved_chunks:
                Retrieved chunks in retrieval order.

        Returns:
            Formatted context string.
        """

        if not retrieved_chunks:
            return ""

        logger.info(
            "Building context for %d retrieved chunk(s).", len(retrieved_chunks)
        )

        parts: list[str] = []

        with Timer() as timer:
            for index, retrieved_chunk in enumerate(retrieved_chunks, start=1):
                parts.append(f"Chunk {index}")
                parts.append(retrieved_chunk.chunk.text)

            context = "\n\n".join(parts)

        logger.info("Built context in %.3f second(s).", timer.elapsed)
        logger.info("Context length: %d characters", len(context))

        return context
