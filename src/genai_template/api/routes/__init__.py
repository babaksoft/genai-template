from genai_template.api.routes.answer import router as answer_router
from genai_template.api.routes.health import router as health_router
from genai_template.api.routes.sources import router as sources_router

__all__ = [
    "answer_router",
    "health_router",
    "sources_router",
]
