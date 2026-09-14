from httpx import get, post, put

from genai_template.config import RagConfig, settings
from genai_template.schemas import (
    AnswerResponse,
    ExperimentResponse,
    IndexBuildResponse,
    RagConfigResponse,
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

    def answer(
        self, query: str, experiment_id: int, rag_config_id: int
    ) -> AnswerResponse:
        """Submit a question to the RAG API.

        Args:
            query:
                User question to submit.
            experiment_id:
                Canonical experiment identifier.
            rag_config_id:
                Canonical RAG configuration identifier.

        Returns:
            API response containing the generated answer, metrics, sources, and
            citation warnings.

        Raises:
            httpx.HTTPStatusError:
                If the API returns an unsuccessful HTTP status code.
        """

        response = post(
            f"{self._base_url}{settings.API_URL_PREFIX}/answer",
            json={
                "query": query,
                "experiment_id": experiment_id,
                "rag_config_id": rag_config_id,
            },
            timeout=settings.REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        return AnswerResponse.model_validate(response.json())

    def list_source_candidates(self) -> list[SourceCandidateResponse]:
        """List corpus directories available for registration.

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
        """List registered corpus sources.

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

    def register_source(self, directory: str) -> SourceResponse:
        """Register one selected corpus directory.

        Args:
            directory:
                Immediate corpus directory name under the configured root.

        Returns:
            Registered source metadata.

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

    def rebuild_index(self, source_id: int, rag_config_id: int) -> IndexBuildResponse:
        """Rebuild one source index with a registered configuration.

        Args:
            source_id:
                Identifier of the source to index.
            rag_config_id:
                Identifier of the indexing configuration.

        Returns:
            Transient index build metrics.

        Raises:
            httpx.HTTPStatusError:
                If the API returns an unsuccessful HTTP status code.
        """

        response = put(
            f"{self._base_url}{settings.API_URL_PREFIX}/sources/"
            f"{source_id}/indexes/{rag_config_id}",
            timeout=settings.REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        return IndexBuildResponse.model_validate(response.json())

    def list_experiments(self) -> list[ExperimentResponse]:
        """List registered experiments.

        Returns:
            Registered experiment metadata.

        Raises:
            httpx.HTTPStatusError:
                If the API returns an unsuccessful HTTP status code.
        """

        response = get(
            f"{self._base_url}{settings.API_URL_PREFIX}/experiments",
            timeout=settings.REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return [ExperimentResponse.model_validate(item) for item in response.json()]

    def get_experiment(self, experiment_id: int) -> ExperimentResponse:
        """Get an experiment by its canonical identifier.

        Args:
            experiment_id:
                Canonical experiment identifier.

        Returns:
            Requested experiment metadata.

        Raises:
            httpx.HTTPStatusError:
                If the API returns an unsuccessful HTTP status code.
        """

        response = get(
            f"{self._base_url}{settings.API_URL_PREFIX}/experiments/{experiment_id}",
            timeout=settings.REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return ExperimentResponse.model_validate(response.json())

    def create_experiment(
        self,
        source_id: int,
        name: str,
        description: str | None = None,
    ) -> ExperimentResponse:
        """Create a source-bound experiment.

        Args:
            source_id:
                Registered source identifier.
            name:
                Experiment display name.
            description:
                Optional experiment description.

        Returns:
            Created experiment metadata.

        Raises:
            httpx.HTTPStatusError:
                If the API returns an unsuccessful HTTP status code.
        """

        response = post(
            f"{self._base_url}{settings.API_URL_PREFIX}/experiments",
            json={"source_id": source_id, "name": name, "description": description},
            timeout=settings.REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return ExperimentResponse.model_validate(response.json())

    def list_rag_configs(self) -> list[RagConfigResponse]:
        """List registered immutable RAG configurations.

        Returns:
            Registered configuration metadata.

        Raises:
            httpx.HTTPStatusError:
                If the API returns an unsuccessful HTTP status code.
        """

        response = get(
            f"{self._base_url}{settings.API_URL_PREFIX}/rag-configs",
            timeout=settings.REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return [RagConfigResponse.model_validate(item) for item in response.json()]

    def get_rag_config(self, rag_config_id: int) -> RagConfigResponse:
        """Get a RAG configuration by its canonical identifier.

        Args:
            rag_config_id:
                Canonical RAG configuration identifier.

        Returns:
            Requested immutable configuration.

        Raises:
            httpx.HTTPStatusError:
                If the API returns an unsuccessful HTTP status code.
        """

        response = get(
            f"{self._base_url}{settings.API_URL_PREFIX}/rag-configs/{rag_config_id}",
            timeout=settings.REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return RagConfigResponse.model_validate(response.json())

    def register_rag_config(self, config: RagConfig) -> RagConfigResponse:
        """Idempotently register a resolved RAG configuration.

        Args:
            config:
                Fully resolved RAG configuration.

        Returns:
            Existing or newly registered immutable configuration.

        Raises:
            httpx.HTTPStatusError:
                If the API returns an unsuccessful HTTP status code.
        """

        response = post(
            f"{self._base_url}{settings.API_URL_PREFIX}/rag-configs",
            json=config.model_dump(mode="json"),
            timeout=settings.REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return RagConfigResponse.model_validate(response.json())
