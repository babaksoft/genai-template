"""Composition helpers for configured RAG application services."""

from genai_template.components.context import ContextBuilder
from genai_template.components.prompt import PromptBuilder
from genai_template.config import RagConfig, settings
from genai_template.db import SessionLocal
from genai_template.factories import (
    create_embedder,
    create_llm,
    create_retrieval_pipeline,
    create_vector_store,
)
from genai_template.pipelines import RetrievalPipeline
from genai_template.services.experiment_service import ExperimentService
from genai_template.services.rag_service import RagService
from genai_template.services.source_service import SourceService


def create_rag_service(config: RagConfig) -> RagService:
    """Compose a RAG service from one fully resolved configuration.

    Args:
        config:
            Configuration shared by retrieval, generation, indexing, and
            experiment tracking components.

    Returns:
        Configured RAG service ready to answer queries.
    """

    embedder = create_embedder(config.embedder)

    def retrieval_pipeline_factory(collection_name: str) -> RetrievalPipeline:
        """Create a retrieval pipeline for a source-owned collection.

        Args:
            collection_name:
                Name of the source collection to query.

        Returns:
            Retrieval pipeline using the configured embedder and store.
        """

        store_config = config.vector_store.model_copy(
            update={"collection_name": collection_name}
        )
        return create_retrieval_pipeline(
            config.retrieval,
            embedder,
            create_vector_store(store_config),
        )

    source_service = SourceService(
        session_factory=SessionLocal,
        corpora_dir=settings.CORPORA_DIR,
        config=config,
    )
    return RagService(
        retrieval_pipeline_factory=retrieval_pipeline_factory,
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        language_model=create_llm(config.llm),
        experiment_service=ExperimentService(SessionLocal),
        source_service=source_service,
        config=config,
    )
