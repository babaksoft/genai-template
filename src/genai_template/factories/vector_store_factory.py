"""Factory for configured vector stores."""

from genai_template.config import settings
from genai_template.config.rag import AnyVectorStoreConfig
from genai_template.protocols import VectorStore
from genai_template.stores.vector import ChromaStore, QdrantStore


def create_vector_store(
    config: AnyVectorStoreConfig,
    collection_name: str,
) -> VectorStore:
    """Create a vector store from validated configuration.

    Args:
        config:
            Vector store configuration.
        collection_name:
            Runtime name of the collection to open or create.

    Returns:
        Configured vector store.

    Raises:
        ValueError:
            If the vector store provider is unsupported.
    """

    if config.type == "chroma":
        return ChromaStore(
            persist_directory=config.persist_directory,
            collection_name=collection_name,
            distance=config.distance,
        )

    if config.type == "qdrant":
        return QdrantStore(
            collection_name=collection_name,
            distance=config.distance,
            path=config.path,
            url=config.url,
            api_key=settings.QDRANT_API_KEY if config.location == "server" else None,
        )

    raise ValueError(f"Unsupported vector store type: {config.type}")
