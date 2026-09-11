"""Factory for configured document splitters."""

from genai_template.components.splitters import DocumentSplitter
from genai_template.config.rag import SplitterConfig
from genai_template.protocols import Splitter


def create_splitter(config: SplitterConfig) -> Splitter:
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

    raise ValueError(f"Unsupported splitter type: {config.type}")
