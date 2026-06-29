"""Pipeline responsible for document ingestion."""

from __future__ import annotations

import logging
from pathlib import Path

from genai_template.components.readers.text_reader import TextReader
from genai_template.components.splitters.sentence_splitter import (
    DocumentSplitter,
)
from genai_template.schemas.chunk import DocumentChunk

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates document ingestion."""

    def __init__(self) -> None:
        """Initialize the ingestion pipeline."""
        self._reader = TextReader()
        self._splitter = DocumentSplitter()

    def run(self, data_dir: Path) -> list[DocumentChunk]:
        """Run the ingestion pipeline.

        Args:
            data_dir:
                Directory containing input documents.

        Returns:
            Canonical document chunks.
        """
        logger.info("Starting ingestion pipeline.")

        documents = self._reader.load(data_dir)
        chunks = self._splitter.split(documents)

        logger.info(
            "Ingestion pipeline completed with %d chunk(s).",
            len(chunks),
        )

        return chunks
