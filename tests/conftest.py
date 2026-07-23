import pytest
from fastapi.testclient import TestClient

from genai_template.api.main import app


@pytest.fixture
def client() -> TestClient:
    """
    Provide a test client for API tests.

    Returns:
        FastAPI test client.
    """

    return TestClient(app)
