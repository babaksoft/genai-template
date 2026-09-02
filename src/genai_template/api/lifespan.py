import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from genai_template.config.logging import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle events.

    Args:
        app:
            FastAPI application instance.
    """

    configure_logging()
    logger.info("Starting GenAI Template API")

    yield

    logger.info("Stopping GenAI Template API")
    tracer_provider = getattr(app.state, "tracer_provider", None)
    if tracer_provider is not None:
        tracer_provider.shutdown()
