from genai_template.api.routes.answer import router as answer_router
from genai_template.api.routes.experiments import router as experiments_router
from genai_template.api.routes.health import router as health_router
from genai_template.api.routes.rag_configs import router as rag_configs_router
from genai_template.api.routes.sources import router as sources_router

__all__ = [
    "answer_router",
    "experiments_router",
    "health_router",
    "rag_configs_router",
    "sources_router",
]
