"""Shared LlamaIndex structured-generation behavior."""

from __future__ import annotations

import logging
from typing import Any, TypeVar, cast

from llama_index.core.output_parsers import PydanticOutputParser
from pydantic import ValidationError

from genai_template.workflow.portfolio.domain.errors import StructuredGenerationError
from genai_template.workflow.portfolio.domain.generation import (
    GenerationRequest,
    ProviderAuditMetadata,
    TokenUsage,
)
from genai_template.workflow.portfolio.domain.summaries import StructuredSummary
from genai_template.workflow.portfolio.ports.structured_generator import (
    StructuredGenerationResponse,
)

logger = logging.getLogger(__name__)
SummaryT = TypeVar("SummaryT", bound=StructuredSummary)


class LlamaIndexStructuredSummaryGenerator[SummaryT: StructuredSummary]:
    """Base adapter that validates LlamaIndex completion text with Pydantic.

    Attributes:
        provider:
            Stable provider identifier.
    """

    provider: str

    def __init__(self, *, llm: Any, model: str) -> None:
        """Initialize the adapter around a configured LlamaIndex LLM.

        Args:
            llm:
                LlamaIndex completion-capable provider object.
            model:
                Provider model identity.
        """

        self._llm = llm
        self._model = model

    def generate(
        self,
        request: GenerationRequest,
        output_type: type[SummaryT],
    ) -> StructuredGenerationResponse[SummaryT]:
        """Generate one summary and normalize provider boundary failures.

        Args:
            request:
                Fully identified generation request.
            output_type:
                Strict summary model used to parse provider JSON.

        Returns:
            Validated summary and non-secret provider accounting.

        Raises:
            StructuredGenerationError:
                If identities mismatch, the provider fails, or output is invalid.
        """

        self._validate_request_identity(request)
        parser = PydanticOutputParser(output_cls=output_type)
        formatted_prompt = parser.format(request.prompt)
        logger.info(
            "Generating %s artifact %s with provider %s and model %s",
            request.provenance.artifact_kind,
            request.provenance.generation_fingerprint,
            self.provider,
            self._model,
        )

        try:
            completion = self._llm.complete(formatted_prompt)
        except Exception:  # noqa: BLE001 - normalize the provider boundary.
            raise StructuredGenerationError(
                "structured generation provider call failed",
                provider=self.provider,
                model=self._model,
                reason="provider-failure",
            ) from None

        try:
            value = cast(SummaryT, parser.parse(str(completion.text)))
        except (ValidationError, ValueError, TypeError):
            raise StructuredGenerationError(
                "provider returned invalid structured output",
                provider=self.provider,
                model=self._model,
                reason="invalid-output",
            ) from None

        usage, metadata = self._extract_accounting(completion)
        return StructuredGenerationResponse(
            value=value,
            provider=self.provider,
            model=self._model,
            token_usage=usage,
            provider_metadata=metadata,
        )

    def _validate_request_identity(self, request: GenerationRequest) -> None:
        """Ensure request provenance agrees with the concrete adapter.

        Args:
            request:
                Request whose provider identity is checked.

        Raises:
            StructuredGenerationError:
                If provider or model provenance does not match this adapter.
        """

        provenance = request.provenance
        if provenance.provider != self.provider or provenance.model != self._model:
            raise StructuredGenerationError(
                "generation request provider identity does not match adapter",
                provider=self.provider,
                model=self._model,
                reason="identity-mismatch",
            )

    def _extract_accounting(
        self, completion: Any
    ) -> tuple[TokenUsage, ProviderAuditMetadata]:
        """Extract provider-specific usage and safe audit metadata.

        Args:
            completion:
                LlamaIndex completion response.

        Returns:
            Token usage and narrow provider audit metadata.
        """

        return TokenUsage(), ProviderAuditMetadata()


def read_value(value: object, name: str) -> object | None:
    """Read a field from either an object or a provider mapping.

    Args:
        value:
            Provider response object or mapping.
        name:
            Field name to retrieve.

    Returns:
        Field value when present, otherwise null.
    """

    if isinstance(value, dict):
        return value.get(name)

    return getattr(value, name, None)


def optional_non_negative_int(value: object | None) -> int | None:
    """Accept provider token counts only when they are non-negative integers.

    Args:
        value:
            Provider usage value.

    Returns:
        A valid token count, otherwise null.
    """

    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else None
    )


def optional_string(value: object | None) -> str | None:
    """Accept provider metadata only when it is a non-empty string.

    Args:
        value:
            Provider metadata value.

    Returns:
        Non-empty string, otherwise null.
    """

    return value if isinstance(value, str) and value else None
