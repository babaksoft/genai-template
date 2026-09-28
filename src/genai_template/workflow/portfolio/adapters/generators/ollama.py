"""Ollama plain-text generator backed by the official Python SDK."""

from __future__ import annotations

import httpx
from ollama import ChatResponse, Client, RequestError, ResponseError

from genai_template.config.ollama import resolve_ollama_base_url
from genai_template.workflow.portfolio.adapters.generators.base import (
    TextGeneratorBase,
    optional_non_negative_int,
)
from genai_template.workflow.portfolio.domain.generation import (
    GenerationRequest,
    ProviderAuditMetadata,
    TokenUsage,
)
from genai_template.workflow.portfolio.ports.text_generator import (
    TextGenerationResponse,
)


class OllamaTextGenerator(TextGeneratorBase):
    """Generate plain Markdown with a configured Ollama model.

    Attributes:
        provider:
            Stable Ollama provider identifier.
    """

    provider = "ollama"

    def __init__(
        self,
        model: str,
        *,
        temperature: float,
        seed: int | None,
        timeout_seconds: float,
        base_url: str | None = None,
        client: Client | None = None,
    ) -> None:
        """Initialize an official Ollama SDK client for local or Cloud use.

        The SDK reads ``OLLAMA_API_KEY`` for Cloud authorization.

        Args:
            model:
                Ollama model name.
            temperature:
                Content-affecting sampling temperature.
            seed:
                Optional deterministic provider seed.
            timeout_seconds:
                Provider request timeout.
            base_url:
                Optional local or Cloud Ollama endpoint override.
            client:
                Optional injected SDK client for deterministic tests.
        """

        super().__init__(model=model)
        self._options: dict[str, object] = {"temperature": temperature}
        if seed is not None:
            self._options["seed"] = seed
        self._client = client or Client(
            host=base_url or resolve_ollama_base_url(),
            timeout=timeout_seconds,
        )

    def generate(self, request: GenerationRequest) -> TextGenerationResponse:
        """Make exactly one non-streaming chat request and validate plain text.

        Args:
            request:
                Fully identified prompt request.

        Returns:
            Complete response text and provider accounting.

        Raises:
            TextGenerationError:
                If identity, transport, completion, or text is invalid.
        """

        self._validate_request_identity(request)
        try:
            response = self._client.chat(
                model=self._model,
                messages=({"role": "user", "content": request.prompt},),
                stream=False,
                options=self._options,
            )
        except (TimeoutError, httpx.TimeoutException):
            self._raise("generation provider request timed out", "provider-timeout")
        except (RequestError, ResponseError, ConnectionError):
            self._raise("generation provider call failed", "provider-failure")
        except Exception:  # noqa: BLE001 - normalize injected/provider transports.
            self._raise("generation provider call failed", "provider-failure")

        if not isinstance(response, ChatResponse):
            self._raise(
                "generation provider returned an invalid response",
                "provider-failure",
            )
        self._validate_ollama_response_model(response.model)
        if response.done_reason in {"content_filter", "refusal"}:
            self._raise("generation provider refused the request", "refusal")
        if response.done is not True or response.done_reason in {
            "length",
            "max_tokens",
        }:
            self._raise("generation provider returned incomplete output", "incomplete")

        text = response.message.content or ""
        if not text.strip():
            self._raise("generation provider returned empty text", "empty-output")

        return TextGenerationResponse(
            text=text,
            provider=self.provider,
            model=self._model,
            token_usage=TokenUsage(
                input_tokens=optional_non_negative_int(response.prompt_eval_count),
                output_tokens=optional_non_negative_int(response.eval_count),
            ),
            provider_metadata=ProviderAuditMetadata(
                finish_reason=response.done_reason,
            ),
        )

    def _validate_ollama_response_model(self, response_model: object) -> None:
        """Accept only the configured model or Ollama's exact Cloud alias form.

        Ollama's authenticated local service accepts a model ending in ``-cloud``
        but attributes the completed response to the same model name without that
        transport-selection suffix. Other model-name differences still fail closed.

        Args:
            response_model:
                Model identity returned by the Ollama service.

        Raises:
            TextGenerationError:
                If the response cannot be attributed to the configured model.
        """

        cloud_model = (
            self._model.removesuffix("-cloud")
            if self._model.endswith("-cloud")
            else None
        )
        if response_model == cloud_model:
            return
        self._validate_response_model(response_model)
