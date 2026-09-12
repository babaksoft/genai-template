"""Embedding model implementations."""

from genai_template.components.embeddings.fastembed import (
    FastEmbedEmbeddingModel,
)
from genai_template.components.embeddings.openai import (
    OpenAIEmbeddingModel,
)

__all__ = [
    "FastEmbedEmbeddingModel",
    "OpenAIEmbeddingModel",
]
