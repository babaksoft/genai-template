import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from genai_template.api.main import create_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """
    Provide a test client for API tests.

    Returns:
        FastAPI test client.
    """

    return TestClient(app)


@pytest.fixture
def app() -> FastAPI:
    """
    Provide a FastAPI application without Phoenix instrumentation.

    Returns:
        FastAPI application.
    """

    return create_app(phoenix_enabled=False)
