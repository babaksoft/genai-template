from fastapi import FastAPI

from genai_template.api.lifespan import lifespan
from genai_template.api.observability import initialize_observability
from genai_template.api.routes import (
    answer_router,
    health_router,
    sources_router,
)
from genai_template.config import settings

app = FastAPI(
    title="GenAI Template API",
    description="API for the GenAI Template RAG experimentation framework.",
    version="0.1.0",
    lifespan=lifespan,
)

app.state.tracer_provider = initialize_observability(app)

app.include_router(
    health_router,
    prefix=settings.API_URL_PREFIX,
)

app.include_router(
    answer_router,
    prefix=settings.API_URL_PREFIX,
)

app.include_router(
    sources_router,
    prefix=settings.API_URL_PREFIX,
)
