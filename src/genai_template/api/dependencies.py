from genai_template.components.context import ContextBuilder
from genai_template.components.prompt import PromptBuilder
from genai_template.config import settings
from genai_template.db import SessionLocal
from genai_template.services import (
    ExperimentService,
    RagConfigService,
    RagService,
    SourceService,
)


def get_rag_service() -> RagService:
    """Provide the application's RAG service.

    Returns:
        Configured RAG service instance.
    """

    rag_config_service = RagConfigService(SessionLocal)
    source_service = SourceService(
        session_factory=SessionLocal,
        corpora_dir=settings.CORPORA_DIR,
        rag_config_service=rag_config_service,
    )
    return RagService(
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        experiment_service=ExperimentService(SessionLocal),
        rag_config_service=rag_config_service,
        source_service=source_service,
    )


def get_source_service() -> SourceService:
    """Provide the application's corpus source service.

    Returns:
        Configured source service.
    """

    return SourceService(
        session_factory=SessionLocal,
        corpora_dir=settings.CORPORA_DIR,
        rag_config_service=RagConfigService(SessionLocal),
    )


def get_experiment_service() -> ExperimentService:
    """Provide the experiment registry service.

    Returns:
        Configured experiment registry.
    """

    return ExperimentService(SessionLocal)


def get_rag_config_service() -> RagConfigService:
    """Provide the RAG configuration registry service.

    Returns:
        Configured RAG configuration registry.
    """

    return RagConfigService(SessionLocal)
