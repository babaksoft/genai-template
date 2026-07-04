"""Context builder component."""

from genai_template.schemas.retrieved_chunk import RetrievedChunk


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

        parts: list[str] = []

        for index, retrieved_chunk in enumerate(retrieved_chunks, start=1):
            parts.append(f"Chunk {index}")
            parts.append(retrieved_chunk.chunk.text)

        return "\n\n".join(parts)
