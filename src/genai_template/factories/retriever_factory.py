"""Factory for configured retrieval pipelines."""

from genai_template.config.rag import RetrievalConfig
from genai_template.pipelines import RetrievalPipeline
from genai_template.protocols import Embedder, Retriever, VectorStore


def create_retrieval_pipeline(
    config: RetrievalConfig,
    embedder: Embedder,
    store: VectorStore,
) -> Retriever:
    """Create a retrieval pipeline from configured components.

    Args:
        config:
            Retrieval configuration.
        embedder:
            Embedding model used for queries.
        store:
            Vector store to search.

    Returns:
        Configured retrieval pipeline.
    """

    return RetrievalPipeline(
        embedder=embedder,
        store=store,
        top_k=config.top_k,
    )
