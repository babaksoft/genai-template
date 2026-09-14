"""API tests for experiment and RAG configuration registries."""

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import Mock

import httpx
from fastapi import FastAPI

from genai_template.api.dependencies import (
    get_experiment_service,
    get_rag_config_service,
)
from genai_template.api.lifespan import lifespan
from genai_template.config import (
    canonical_config_json,
    config_fingerprint,
    load_rag_config,
)
from genai_template.db.models import Experiment
from genai_template.db.models import RagConfig as RagConfigRecord
from genai_template.services import ExperimentService, RagConfigService


async def request(
    app: FastAPI,
    method: str,
    path: str,
    json: dict[str, Any] | None = None,
) -> httpx.Response:
    """Send a request directly to the ASGI application.

    Args:
        app:
            FastAPI test application.
        method:
            HTTP request method.
        path:
            API path to request.
        json:
            Optional JSON request body.

    Returns:
        HTTP response from the application.
    """

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.request(method, path, json=json)


def dependency_override(value: Any) -> Any:
    """Create an asynchronous FastAPI dependency override.

    Args:
        value:
            Value returned by the dependency.

    Returns:
        Async dependency function returning the supplied value.
    """

    async def override() -> Any:
        """Return the captured dependency value.

        Returns:
            Captured dependency value.
        """

        return value

    return override


async def enter_lifespan(app: FastAPI) -> None:
    """Enter and exit the application lifespan.

    Args:
        app:
            FastAPI test application.
    """

    async with lifespan(app):
        pass


def test_experiment_endpoints_create_list_and_get(app: FastAPI) -> None:
    """Experiment endpoints should expose database-ID-based registry operations.

    Args:
        app:
            FastAPI test application.
    """

    created_at = datetime(2026, 9, 13, tzinfo=UTC)
    experiment = Experiment(
        id=3,
        source_id=2,
        name="trial",
        description="Compare prompts",
        created_at=created_at,
    )
    service = Mock(spec=ExperimentService)
    service.create_experiment.return_value = experiment
    service.list_experiments.return_value = [experiment]
    service.get_experiment.return_value = experiment
    app.dependency_overrides[get_experiment_service] = dependency_override(service)
    created = asyncio.run(
        request(
            app,
            "POST",
            "/api/v1/experiments",
            {"source_id": 2, "name": "trial", "description": "Compare prompts"},
        )
    )
    listed = asyncio.run(request(app, "GET", "/api/v1/experiments"))
    fetched = asyncio.run(request(app, "GET", "/api/v1/experiments/3"))

    assert created.status_code == 201
    assert created.json()["id"] == 3
    assert listed.json() == [created.json()]
    assert fetched.json() == created.json()
    service.create_experiment.assert_called_once_with(
        source_id=2, name="trial", description="Compare prompts"
    )
    service.get_experiment.assert_called_once_with(3)
    app.dependency_overrides.clear()


def test_experiment_creation_rejects_missing_source(app: FastAPI) -> None:
    """Experiment creation should expose missing sources as HTTP 404.

    Args:
        app:
            FastAPI test application.
    """

    service = Mock(spec=ExperimentService)
    service.create_experiment.side_effect = ValueError("Source 9 does not exist.")
    app.dependency_overrides[get_experiment_service] = dependency_override(service)

    response = asyncio.run(
        request(
            app,
            "POST",
            "/api/v1/experiments",
            {"source_id": 9, "name": "trial"},
        )
    )

    assert response.status_code == 404
    app.dependency_overrides.clear()


def test_rag_config_endpoints_register_list_and_get(app: FastAPI) -> None:
    """Config endpoints should return validated configuration snapshots.

    Args:
        app:
            FastAPI test application.
    """

    config = load_rag_config()
    record = RagConfigRecord(
        id=4,
        config_fingerprint=config_fingerprint(config),
        config_json=canonical_config_json(config),
        created_at=datetime(2026, 9, 13, tzinfo=UTC),
    )
    service = Mock(spec=RagConfigService)
    service.register_config.return_value = record
    service.list_configs.return_value = [record]
    service.get_config.return_value = record
    app.dependency_overrides[get_rag_config_service] = dependency_override(service)
    created = asyncio.run(
        request(
            app,
            "POST",
            "/api/v1/rag-configs",
            config.model_dump(mode="json"),
        )
    )
    listed = asyncio.run(request(app, "GET", "/api/v1/rag-configs"))
    fetched = asyncio.run(request(app, "GET", "/api/v1/rag-configs/4"))

    assert created.status_code == 201
    assert created.json()["id"] == 4
    assert created.json()["config_fingerprint"] == config_fingerprint(config)
    assert listed.json() == [created.json()]
    assert fetched.json() == created.json()
    service.get_config.assert_called_once_with(4)
    app.dependency_overrides.clear()


def test_registry_get_endpoints_return_not_found(app: FastAPI) -> None:
    """Registry lookups should translate missing IDs to HTTP 404.

    Args:
        app:
            FastAPI test application.
    """

    experiments = Mock(spec=ExperimentService)
    experiments.get_experiment.side_effect = ValueError("Experiment 7 does not exist.")
    configs = Mock(spec=RagConfigService)
    configs.get_config.side_effect = ValueError("RAG config 8 does not exist.")
    app.dependency_overrides[get_experiment_service] = dependency_override(experiments)
    app.dependency_overrides[get_rag_config_service] = dependency_override(configs)
    experiment_response = asyncio.run(request(app, "GET", "/api/v1/experiments/7"))
    config_response = asyncio.run(request(app, "GET", "/api/v1/rag-configs/8"))

    assert experiment_response.status_code == 404
    assert config_response.status_code == 404
    app.dependency_overrides.clear()
