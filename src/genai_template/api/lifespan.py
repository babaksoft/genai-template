import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle events.

    Args:
        app:
            FastAPI application instance.
    """

    logger.info("Starting GenAI Template API")

    yield

    logger.info("Stopping GenAI Template API")
