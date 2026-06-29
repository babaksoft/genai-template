"""Reader implementation for loading text-based documents."""

from __future__ import annotations

import logging
from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader

logger = logging.getLogger(__name__)


class TextReader:
    """Loads text-based documents from a directory."""

    def load(self, directory: Path) -> list[Document]:
        """Load documents from a directory.

        Args:
            directory:
                Directory containing input documents.

        Returns:
            A list of LlamaIndex ``Document`` objects.

        Raises:
            FileNotFoundError:
                If the directory does not exist.
            NotADirectoryError:
                If the supplied path is not a directory.
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory does not exist: {directory}")

        if not directory.is_dir():
            raise NotADirectoryError(f"Expected a directory: {directory}")

        logger.info("Loading documents from '%s'.", directory)
        supported_files = [
            path
            for extension in ("*.md", "*.txt")
            for path in directory.glob(extension)
        ]

        if not supported_files:
            logger.info("No supported documents found in '%s'.", directory)
            return []

        documents = SimpleDirectoryReader(
            input_dir=str(directory),
            required_exts=[".md", ".txt"],
            filename_as_id=True,
        ).load_data()

        logger.info("Loaded %d document(s).", len(documents))

        return documents
