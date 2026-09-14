"""Opt-in OpenTelemetry tracing for the API."""

import logging

from fastapi import FastAPI
from opentelemetry.sdk.trace import TracerProvider

from genai_template.config import settings

logger = logging.getLogger(__name__)


def initialize_observability(app: FastAPI) -> TracerProvider | None:
    """Configure Phoenix request tracing when it is explicitly enabled."""

    phoenix_enabled = getattr(app.state, "phoenix_enabled", False)
    if not phoenix_enabled:
        return None

    # Imported only when tracing is enabled so the default API startup has no
    # Phoenix setup side effects.
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from phoenix.otel import register

    tracer_provider = register(
        project_name=settings.PHOENIX_PROJECT_NAME,
        endpoint=settings.PHOENIX_COLLECTOR_ENDPOINT,
        batch=True,
    )
    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=tracer_provider,
    )
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    logger.info("Phoenix tracing enabled for project %s", settings.PHOENIX_PROJECT_NAME)

    return tracer_provider
