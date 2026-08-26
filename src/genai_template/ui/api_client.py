from httpx import post

from genai_template.config import settings
from genai_template.schemas import AnswerResponse


class ApiClient:
    """Client for communicating with the GenAI Template API."""

    def __init__(self, base_url: str) -> None:
        """Initialize the API client.

        Args:
            base_url:
                Base URL of the GenAI Template API.
        """

        self._base_url = base_url.rstrip("/")

    def answer(self, query: str) -> AnswerResponse:
        """Submit a question to the RAG API.

        Args:
            query:
                User question to submit.

        Returns:
            API response containing the generated answer and metrics.

        Raises:
            httpx.HTTPStatusError:
                If the API returns an unsuccessful HTTP status code.
        """

        response = post(
            f"{self._base_url}{settings.API_URL_PREFIX}/answer",
            json={"query": query},
            timeout=settings.REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        return AnswerResponse.model_validate(response.json())
