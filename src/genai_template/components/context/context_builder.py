"""Context builder component."""

import json
import logging
from pathlib import PurePosixPath
from typing import Any

from genai_template.components.readers.portfolio_metadata import (
    COMPONENT_ID,
    CORPUS_FINGERPRINT,
    DOCUMENT_TYPE,
    GENERATION_FINGERPRINT,
    PROJECT_DISPLAY_NAME,
    PROJECT_SLUG,
    REPOSITORY_URL,
    RESOLVED_COMMIT_SHA,
    has_portfolio_metadata,
    validate_portfolio_metadata,
)
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
            portfolio_source_count = 0
            project_slugs: set[str] = set()

            with Timer() as timer:
                for index, retrieved_chunk in enumerate(retrieved_chunks, start=1):
                    chunk = retrieved_chunk.chunk
                    provenance = self._portfolio_provenance(chunk.metadata)
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
                        **provenance,
                    )
                    sources.append(source)

                    block = [f"[{label}]", f"Document: {document_name}"]
                    if source.project_slug is not None:
                        portfolio_source_count += 1
                        project_slugs.add(source.project_slug)
                        block.append(
                            "Project: "
                            f"{source.project_display_name} ({source.project_slug})"
                        )
                        if source.repository_url is not None:
                            block.append(f"Repository: {source.repository_url}")
                        block.append(f"Revision: {source.resolved_commit_sha}")
                        block.append(f"Document type: {source.document_type}")
                        if source.component_id is not None:
                            block.append(f"Component: {source.component_id}")
                    if section is not None:
                        block.append(f"Section: {section}")
                    block.extend(["Content:", chunk.text])
                    parts.append("\n".join(block))

                context = "\n\n".join(parts)
            span.set_attribute(OUTPUT_VALUE, context)
            if portfolio_source_count:
                span.set_attribute(
                    "rag.context.portfolio_source_count", portfolio_source_count
                )
                span.set_attribute("rag.context.project_count", len(project_slugs))
                span.set_attribute(
                    "rag.context.project_slugs", json.dumps(sorted(project_slugs))
                )

        logger.info("Built context in %.3f second(s).", timer.elapsed)
        logger.info("Context length: %d characters", len(context))

        return CitationContext(text=context, sources=sources)

    @staticmethod
    def _portfolio_provenance(metadata: dict[str, Any]) -> dict[str, str | None]:
        """Extract a complete validated Portfolio provenance projection.

        Generic metadata may omit the entire Portfolio provenance group. Once any
        Portfolio-specific key is present, the complete controlled contract is
        required so citations cannot imply incomplete provenance.

        Args:
            metadata:
                Retrieved chunk metadata.

        Returns:
            Citation fields for a Portfolio chunk, or an empty mapping for a
            generic chunk.

        Raises:
            ValueError:
                If claimed Portfolio metadata is incomplete or malformed.
        """

        if not has_portfolio_metadata(metadata):
            return {}

        validate_portfolio_metadata(metadata, allow_header_path=True)
        return {
            "project_slug": metadata[PROJECT_SLUG],
            "project_display_name": metadata[PROJECT_DISPLAY_NAME],
            "document_type": metadata[DOCUMENT_TYPE],
            "component_id": metadata.get(COMPONENT_ID),
            "repository_url": metadata.get(REPOSITORY_URL),
            "resolved_commit_sha": metadata[RESOLVED_COMMIT_SHA],
            "corpus_fingerprint": metadata[CORPUS_FINGERPRINT],
            "generation_fingerprint": metadata[GENERATION_FINGERPRINT],
        }

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
