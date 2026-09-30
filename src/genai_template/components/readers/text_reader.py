"""Reader implementation for loading text-based documents."""

from __future__ import annotations

import logging
from pathlib import Path

from llama_index.core import SimpleDirectoryReader

from genai_template.schemas.corpus import LoadedDocuments
from genai_template.utils import Timer

logger = logging.getLogger(__name__)


class TextReader:
    """Loads text-based documents from a directory."""

    def __init__(self, *, exclude_hidden: bool = True) -> None:
        """Initialize the text reader.

        Args:
            exclude_hidden:
                Whether the underlying reader should reject hidden paths. Portfolio
                releases disable this after manifest validation because their
                immutable parent directory is intentionally hidden.
        """

        self._exclude_hidden = exclude_hidden

    def load(self, directory: Path) -> LoadedDocuments:
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

        with Timer() as timer:
            supported_files = [
                path
                for pattern in ("*.md", "*.txt")
                for path in directory.glob(pattern)
            ]

            if not supported_files:
                logger.info("No supported documents found in '%s'.", directory)
                return LoadedDocuments(documents=())

            documents = tuple(
                SimpleDirectoryReader(
                    input_dir=str(directory),
                    required_exts=[".md", ".txt"],
                    filename_as_id=True,
                    exclude_hidden=self._exclude_hidden,
                ).load_data()
            )

        logger.info(
            "Loaded %d document(s) in %.3f second(s).",
            len(documents),
            timer.elapsed,
        )

        return LoadedDocuments(documents=documents)
