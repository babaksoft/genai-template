"""Context builder component."""

import logging
from pathlib import PurePosixPath

from genai_template.observability import INPUT_VALUE, OUTPUT_VALUE, application_span
from genai_template.schemas import CitationContext, CitationSource, RetrievedChunk
from genai_template.utils import Timer

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Build citation-aware LLM context from retrieved chunks."""

    def build(self, retrieved_chunks: list[RetrievedChunk]) -> CitationContext:
        """
        Build a context string from retrieved chunks.

        Args:
            retrieved_chunks:
                Retrieved chunks in retrieval order.

        Returns:
            Formatted context and its ordered citation sources.
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
                return CitationContext(text="", sources=[])

            logger.info(
                "Building context for %d retrieved chunk(s).", len(retrieved_chunks)
            )

            parts: list[str] = []
            sources: list[CitationSource] = []

            with Timer() as timer:
                for index, retrieved_chunk in enumerate(retrieved_chunks, start=1):
                    chunk = retrieved_chunk.chunk
                    label = f"S{index}"
                    document_name = self._document_name(
                        chunk.metadata.get("file_name"), chunk.document_id
                    )
                    section_value = chunk.metadata.get("header_path")
                    section = (
                        section_value
                        if isinstance(section_value, str) and section_value
                        else None
                    )
                    source = CitationSource(
                        label=label,
                        chunk_id=chunk.id,
                        document_name=document_name,
                        section=section,
                        content=chunk.text,
                        distance=retrieved_chunk.distance,
                        cited=False,
                    )
                    sources.append(source)

                    block = [f"[{label}]", f"Document: {document_name}"]
                    if section is not None:
                        block.append(f"Section: {section}")
                    block.extend(["Content:", chunk.text])
                    parts.append("\n".join(block))

                context = "\n\n".join(parts)
            span.set_attribute(OUTPUT_VALUE, context)

        logger.info("Built context in %.3f second(s).", timer.elapsed)
        logger.info("Context length: %d characters", len(context))

        return CitationContext(text=context, sources=sources)

    @staticmethod
    def _document_name(file_name: object, document_id: str) -> str:
        """Return a safe basename without consulting absolute file paths.

        Args:
            file_name:
                Optional file-name metadata value.
            document_id:
                Parent document identifier used as a fallback.

        Returns:
            Safe document basename.
        """

        candidate = (
            file_name if isinstance(file_name, str) and file_name else document_id
        )
        return PurePosixPath(candidate.replace("\\", "/")).name
