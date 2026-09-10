"""Factory for configured vector stores."""

from genai_template.config.rag import VectorStoreConfig
from genai_template.stores.vector import ChromaStore


def create_vector_store(config: VectorStoreConfig) -> ChromaStore:
    """Create a vector store from validated configuration.

    Args:
        config:
            Vector store configuration.

    Returns:
        Configured vector store.

    Raises:
        ValueError:
            If the vector store provider is unsupported.
    """

    if config.type == "chroma":
        return ChromaStore(
            persist_directory=config.persist_directory,
            collection_name=config.collection_name,
            distance=config.distance,
        )

    raise ValueError(f"Unsupported vector store type: {config.type}")
