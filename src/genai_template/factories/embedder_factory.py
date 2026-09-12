"""Factory for configured embedding models."""

from genai_template.components.embeddings import FastEmbedEmbeddingModel
from genai_template.config.rag import AnyEmbedderConfig
from genai_template.protocols import Embedder


def create_embedder(config: AnyEmbedderConfig) -> Embedder:
    """Create an embedding model from validated configuration.

    Args:
        config:
            Embedder configuration.

    Returns:
        Configured embedding model.

    Raises:
        ValueError:
            If the embedding provider is unsupported.
    """

    if config.type == "fastembed":
        return FastEmbedEmbeddingModel(model_name=config.model_name)

    raise ValueError(f"Unsupported embedder type: {config.type}")
