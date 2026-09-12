"""Document splitter built on top of LlamaIndex's MarkdownNodeParser."""

from __future__ import annotations

import logging

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser

from genai_template.schemas import DocumentChunk
from genai_template.utils import Timer

logger = logging.getLogger(__name__)


class MarkdownDocumentSplitter:
    """Split Markdown documents into header-based canonical chunks."""

    def __init__(self, header_path_separator: str = "/") -> None:
        """Initialize the Markdown document splitter.

        Args:
            header_path_separator:
                Separator used in parser-generated header paths.
        """

        self._splitter = MarkdownNodeParser(
            header_path_separator=header_path_separator,
        )

    def split(self, documents: list[Document]) -> list[DocumentChunk]:
        """Split documents at Markdown headers.

        Args:
            documents:
                Documents to split.

        Returns:
            Header-based chunks with source and header-path metadata.
        """

        if not documents:
            return []

        logger.info("Splitting %d Markdown document(s).", len(documents))

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

                chunks.append(
                    DocumentChunk(
                        id=f"{document_id}-{index:03d}",
                        document_id=document_id,
                        text=node.get_content(),
                        metadata=metadata,
                    )
                )

        logger.info(
            "Generated %d Markdown chunk(s) in %.3f second(s).",
            len(chunks),
            timer.elapsed,
        )

        return chunks
