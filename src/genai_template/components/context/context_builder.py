"""Context builder component."""

import logging

from genai_template.observability import INPUT_VALUE, OUTPUT_VALUE, application_span
from genai_template.schemas import RetrievedChunk
from genai_template.utils import Timer

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

        with application_span(
            "rag.context.build",
            "CHAIN",
            {
                INPUT_VALUE: "\n\n".join(item.chunk.text for item in retrieved_chunks),
                "rag.chunk_count": len(retrieved_chunks),
            },
        ) as span:
            if not retrieved_chunks:
                span.set_attribute(OUTPUT_VALUE, "")
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
            span.set_attribute(OUTPUT_VALUE, context)

        logger.info("Built context in %.3f second(s).", timer.elapsed)
        logger.info("Context length: %d characters", len(context))

        return context
