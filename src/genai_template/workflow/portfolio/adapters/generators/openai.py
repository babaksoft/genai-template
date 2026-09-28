"""OpenAI plain-text generator backed by the official Responses API."""

from __future__ import annotations

from openai import APIError, APITimeoutError, OpenAI
from openai.types.responses import Response

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


class OpenAITextGenerator(TextGeneratorBase):
    """Generate plain Markdown with a configured OpenAI model.

    Attributes:
        provider:
            Stable OpenAI provider identifier.
    """

    provider = "openai"

    def __init__(
        self,
        model: str,
        *,
        temperature: float,
        timeout_seconds: float,
        client: OpenAI | None = None,
    ) -> None:
        """Initialize an official OpenAI SDK client with retries disabled.

        Args:
            model:
                OpenAI model name.
            temperature:
                Content-affecting sampling temperature.
            timeout_seconds:
                Provider request timeout.
            client:
                Optional injected SDK client for deterministic tests.
        """

        super().__init__(model=model)
        self._temperature = temperature
        self._client = client or OpenAI(
            timeout=timeout_seconds,
            max_retries=0,
        )

    def generate(self, request: GenerationRequest) -> TextGenerationResponse:
        """Make exactly one Responses API request and validate plain text.

        Args:
            request:
                Fully identified prompt request.

        Returns:
            Complete response text and provider accounting.

        Raises:
            TextGenerationError:
                If identity, transport, refusal, completion, or text is invalid.
        """

        self._validate_request_identity(request)
        try:
            response = self._client.responses.create(
                model=self._model,
                input=request.prompt,
                temperature=self._temperature,
                store=False,
            )
        except APITimeoutError:
            self._raise("generation provider request timed out", "provider-timeout")
        except APIError:
            self._raise("generation provider call failed", "provider-failure")
        except Exception:  # noqa: BLE001 - normalize injected/provider transports.
            self._raise("generation provider call failed", "provider-failure")

        return self._response(response)

    def _response(self, response: Response) -> TextGenerationResponse:
        """Validate and normalize one SDK response.

        Args:
            response:
                Official SDK Responses API value.

        Returns:
            Validated provider-neutral text response.

        Raises:
            TextGenerationError:
                If the response is refused, incomplete, empty, or misattributed.
        """

        self._validate_response_model(response.model)
        if any(
            content.type == "refusal"
            for item in response.output
            if item.type == "message"
            for content in item.content
        ):
            self._raise("generation provider refused the request", "refusal")
        if response.status == "incomplete":
            self._raise("generation provider returned incomplete output", "incomplete")
        if response.status != "completed":
            self._raise("generation provider did not complete", "provider-failure")

        text = response.output_text
        if not text.strip():
            self._raise("generation provider returned empty text", "empty-output")

        usage = response.usage
        return TextGenerationResponse(
            text=text,
            provider=self.provider,
            model=self._model,
            token_usage=TokenUsage(
                input_tokens=optional_non_negative_int(
                    usage.input_tokens if usage is not None else None
                ),
                output_tokens=optional_non_negative_int(
                    usage.output_tokens if usage is not None else None
                ),
            ),
            provider_metadata=ProviderAuditMetadata(
                request_id=response.id,
                finish_reason=response.status,
            ),
        )
