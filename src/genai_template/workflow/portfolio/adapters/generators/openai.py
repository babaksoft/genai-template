"""OpenAI structured summary adapter backed by LlamaIndex."""

from __future__ import annotations

from typing import Any

from llama_index.llms.openai import OpenAI

from genai_template.workflow.portfolio.adapters.generators.base import (
    LlamaIndexStructuredSummaryGenerator,
    optional_non_negative_int,
    optional_string,
    read_value,
)
from genai_template.workflow.portfolio.domain.generation import (
    ProviderAuditMetadata,
    TokenUsage,
)


class OpenAIStructuredSummaryGenerator(LlamaIndexStructuredSummaryGenerator[Any]):
    """Generate validated summaries with a configured OpenAI model.

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
    ) -> None:
        """Initialize the OpenAI structured generation adapter.

        Args:
            model:
                OpenAI model name.
            temperature:
                Content-affecting sampling temperature.
            timeout_seconds:
                Provider request timeout.
        """

        llm = OpenAI(model=model, temperature=temperature, timeout=timeout_seconds)
        super().__init__(llm=llm, model=model)

    def _extract_accounting(
        self, completion: Any
    ) -> tuple[TokenUsage, ProviderAuditMetadata]:
        """Extract OpenAI token usage and safe response metadata.

        Args:
            completion:
                LlamaIndex OpenAI completion response.

        Returns:
            Token usage and safe audit metadata.
        """

        raw = completion.raw
        usage = read_value(raw, "usage")
        choices = read_value(raw, "choices")
        first_choice = choices[0] if isinstance(choices, list) and choices else None
        return (
            TokenUsage(
                input_tokens=optional_non_negative_int(
                    read_value(usage, "prompt_tokens")
                ),
                output_tokens=optional_non_negative_int(
                    read_value(usage, "completion_tokens")
                ),
            ),
            ProviderAuditMetadata(
                request_id=optional_string(read_value(raw, "id")),
                finish_reason=optional_string(
                    read_value(first_choice, "finish_reason")
                ),
            ),
        )
