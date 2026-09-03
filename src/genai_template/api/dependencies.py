from genai_template.components.context import ContextBuilder
from genai_template.components.embeddings import FastEmbedEmbeddingModel
from genai_template.components.language_models import OllamaLanguageModel
from genai_template.components.prompt import PromptBuilder
from genai_template.config import settings
from genai_template.db import SessionLocal
from genai_template.pipelines import RetrievalPipeline
from genai_template.services import ExperimentService, RagService, SourceService
from genai_template.stores.vector import ChromaStore


def create_retrieval_pipeline() -> RetrievalPipeline:
    """
    Create retrieval pipeline using the active application configuration.

    Returns:
        Configured retrieval pipeline.
    """

    return RetrievalPipeline(
        embedder=FastEmbedEmbeddingModel(),
        store=ChromaStore(
            persist_directory=settings.CHROMA_PERSIST_DIR,
        ),
    )


def create_language_model() -> OllamaLanguageModel:
    """
    Create language model using the active application configuration.

    Returns:
        Configured language model.
    """

    return OllamaLanguageModel(settings.LLM_MODEL)


def create_experiment_service() -> ExperimentService:
    """
    Create experiment service using the active application configuration.

    Returns:
        Configured experiment service.
    """

    return ExperimentService(
        session_factory=SessionLocal,
    )


def get_rag_service() -> RagService:
    """Provide the application's RAG service.

    Returns:
        Configured RAG service instance.
    """

    return RagService(
        retrieval_pipeline=create_retrieval_pipeline(),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        language_model=create_language_model(),
        experiment_service=create_experiment_service(),
    )


def get_source_service() -> SourceService:
    """Provide the application's corpus source service.

    Returns:
        Configured source service.
    """

    return SourceService(
        session_factory=SessionLocal,
        corpora_dir=settings.CORPORA_DIR,
    )
