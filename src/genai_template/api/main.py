"""
The actual application routes that call the services layer.
"""

from fastapi import FastAPI

from genai_template.api.lifespan import lifespan
from genai_template.api.routes import answer, health

app = FastAPI(
    title="GenAI Template API",
    description="API for the GenAI Template RAG experimentation framework.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(
    health.router,
    prefix="/api/v1",
)

app.include_router(
    answer.router,
    prefix="/api/v1",
)
