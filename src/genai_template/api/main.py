from fastapi import FastAPI

from genai_template.api.lifespan import lifespan
from genai_template.api.routes import (
    answer_router,
    health_router,
    sources_router,
)
from genai_template.config import settings


def create_app(*, phoenix_enabled: bool | None = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        phoenix_enabled:
            Whether Phoenix instrumentation should be enabled.
            If omitted, the application configuration is used.

    Returns:
        Configured FastAPI application.
    """

    app = FastAPI(
        title="GenAI Template API",
        description="API for the GenAI Template RAG experimentation framework.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.phoenix_enabled = (
        settings.PHOENIX_ENABLED if phoenix_enabled is None else phoenix_enabled
    )

    app.include_router(health_router, prefix=settings.API_URL_PREFIX)
    app.include_router(answer_router, prefix=settings.API_URL_PREFIX)
    app.include_router(sources_router, prefix=settings.API_URL_PREFIX)

    return app


app = create_app()
