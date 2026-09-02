"""Tests for optional Phoenix API tracing."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI

from genai_template.api.lifespan import lifespan
from genai_template.api.observability import initialize_observability


def test_initialize_observability_skips_disabled_tracing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled tracing must not register a provider or instrument FastAPI."""

    monkeypatch.setattr(
        "genai_template.api.observability.settings.PHOENIX_ENABLED", False
    )

    with (
        patch("phoenix.otel.register") as register,
        patch(
            "opentelemetry.instrumentation.fastapi.FastAPIInstrumentor.instrument_app"
        ) as instrument_app,
        patch(
            "openinference.instrumentation.llama_index.LlamaIndexInstrumentor.instrument"
        ) as instrument_llama_index,
    ):
        tracer_provider = initialize_observability(FastAPI())

    assert tracer_provider is None
    register.assert_not_called()
    instrument_app.assert_not_called()
    instrument_llama_index.assert_not_called()


def test_initialize_observability_configures_phoenix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled tracing uses the configured Phoenix destination and project."""

    app = FastAPI()
    tracer_provider = MagicMock()
    monkeypatch.setattr(
        "genai_template.api.observability.settings.PHOENIX_ENABLED", True
    )
    monkeypatch.setattr(
        "genai_template.api.observability.settings.PHOENIX_COLLECTOR_ENDPOINT",
        "http://localhost:6006/v1/traces",
    )
    monkeypatch.setattr(
        "genai_template.api.observability.settings.PHOENIX_PROJECT_NAME",
        "test-project",
    )

    with (
        patch("phoenix.otel.register", return_value=tracer_provider) as register,
        patch(
            "opentelemetry.instrumentation.fastapi.FastAPIInstrumentor.instrument_app"
        ) as instrument_app,
        patch(
            "openinference.instrumentation.llama_index.LlamaIndexInstrumentor.instrument"
        ) as instrument_llama_index,
    ):
        result = initialize_observability(app)

    assert result is tracer_provider
    register.assert_called_once_with(
        project_name="test-project",
        endpoint="http://localhost:6006/v1/traces",
        batch=True,
    )
    instrument_app.assert_called_once_with(
        app,
        tracer_provider=tracer_provider,
        excluded_urls="/api/v1/ingest",
    )
    instrument_llama_index.assert_called_once_with(tracer_provider=tracer_provider)


def test_lifespan_shuts_down_initialized_tracer_provider() -> None:
    """The API lifespan releases the provider that tracing initialization made."""

    app = FastAPI()
    app.state.tracer_provider = MagicMock()

    async def run_lifespan() -> None:
        async with lifespan(app):
            pass

    asyncio.run(run_lifespan())

    app.state.tracer_provider.shutdown.assert_called_once_with()


def test_lifespan_skips_shutdown_without_tracer_provider() -> None:
    """The API lifespan is safe when observability was disabled."""

    async def run_lifespan() -> None:
        async with lifespan(FastAPI()):
            pass

    asyncio.run(run_lifespan())
