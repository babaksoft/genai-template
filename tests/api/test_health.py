import pytest
from fastapi.testclient import TestClient

from genai_template.api.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_check_returns_healthy_status(
    client: TestClient,
) -> None:
    """Verify the health endpoint returns a healthy status.

    Args:
        client:
            FastAPI test client.
    """

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
    }
