from typing import Annotated

from fastapi import Depends

from genai_template.config import RagConfig, load_rag_config, settings
from genai_template.db import SessionLocal
from genai_template.services import RagService, SourceService, create_rag_service


def get_rag_config() -> RagConfig:
    """Provide the default RAG configuration derived from application settings.

    Returns:
        Fully resolved default RAG configuration.
    """

    return load_rag_config()


def get_rag_service(
    config: Annotated[RagConfig, Depends(get_rag_config)],
) -> RagService:
    """Provide the application's RAG service.

    Args:
        config:
            Default resolved RAG configuration.

    Returns:
        Configured RAG service instance.
    """

    return create_rag_service(config)


def get_source_service(
    config: Annotated[RagConfig, Depends(get_rag_config)],
) -> SourceService:
    """Provide the application's corpus source service.

    Args:
        config:
            Default resolved RAG configuration.

    Returns:
        Configured source service.
    """

    return SourceService(
        session_factory=SessionLocal,
        corpora_dir=settings.CORPORA_DIR,
        config=config,
    )
