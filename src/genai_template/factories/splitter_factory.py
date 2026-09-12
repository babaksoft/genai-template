"""Factory for configured document splitters."""

from genai_template.components.splitters import (
    DocumentSplitter,
    MarkdownDocumentSplitter,
)
from genai_template.config.rag import AnySplitterConfig
from genai_template.protocols import Splitter


def create_splitter(config: AnySplitterConfig) -> Splitter:
    """Create a document splitter from validated configuration.

    Args:
        config:
            Splitter configuration.

    Returns:
        Configured document splitter.

    Raises:
        ValueError:
            If the splitter provider is unsupported.
    """

    if config.type == "sentence":
        return DocumentSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
    if config.type == "markdown":
        return MarkdownDocumentSplitter(
            header_path_separator=config.header_path_separator,
        )

    raise ValueError(f"Unsupported splitter type: {config.type}")
