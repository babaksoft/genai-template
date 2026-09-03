from httpx import get, post

from genai_template.config import settings
from genai_template.schemas import (
    AnswerResponse,
    SourceCandidateResponse,
    SourceResponse,
)


class ApiClient:
    """Client for communicating with the GenAI Template API."""

    def __init__(self, base_url: str) -> None:
        """Initialize the API client.

        Args:
            base_url:
                Base URL of the GenAI Template API.
        """

        self._base_url = base_url.rstrip("/")

    def answer(self, query: str, source_id: int) -> AnswerResponse:
        """Submit a question to the RAG API.

        Args:
            query:
                User question to submit.
            source_id:
                Identifier of the source used for retrieval.

        Returns:
            API response containing the generated answer and metrics.

        Raises:
            httpx.HTTPStatusError:
                If the API returns an unsuccessful HTTP status code.
        """

        response = post(
            f"{self._base_url}{settings.API_URL_PREFIX}/answer",
            json={"query": query, "source_id": source_id},
            timeout=settings.REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        return AnswerResponse.model_validate(response.json())

    def list_source_candidates(self) -> list[SourceCandidateResponse]:
        """List corpus directories available for ingestion.

        Returns:
            Available source directory names.

        Raises:
            httpx.HTTPStatusError:
                If the API returns an unsuccessful HTTP status code.
        """

        response = get(
            f"{self._base_url}{settings.API_URL_PREFIX}/sources/candidates",
            timeout=settings.REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        return [
            SourceCandidateResponse.model_validate(candidate)
            for candidate in response.json()
        ]

    def list_sources(self) -> list[SourceResponse]:
        """List successfully ingested corpus sources.

        Returns:
            Persisted source metadata.

        Raises:
            httpx.HTTPStatusError:
                If the API returns an unsuccessful HTTP status code.
        """

        response = get(
            f"{self._base_url}{settings.API_URL_PREFIX}/sources",
            timeout=settings.REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        return [SourceResponse.model_validate(source) for source in response.json()]

    def ingest_source(self, directory: str) -> SourceResponse:
        """Ingest one selected corpus directory.

        Args:
            directory:
                Immediate corpus directory name under the configured root.

        Returns:
            Ingested source metadata.

        Raises:
            httpx.HTTPStatusError:
                If the API returns an unsuccessful HTTP status code.
        """

        response = post(
            f"{self._base_url}{settings.API_URL_PREFIX}/sources",
            json={"directory": directory},
            timeout=settings.REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        return SourceResponse.model_validate(response.json())
