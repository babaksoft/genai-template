from fastapi import FastAPI

from genai_template.api.lifespan import lifespan
from genai_template.api.routes import answer, health, ingest
from genai_template.config import settings

app = FastAPI(
    title="GenAI Template API",
    description="API for the GenAI Template RAG experimentation framework.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(
    health.router,
    prefix=settings.URL_PREFIX,
)

app.include_router(
    answer.router,
    prefix=settings.URL_PREFIX,
)

app.include_router(
    ingest.router,
    prefix=settings.URL_PREFIX,
)
