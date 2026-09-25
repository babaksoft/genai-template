"""Ollama structured summary adapter backed by LlamaIndex."""

from __future__ import annotations

from typing import Any

from llama_index.llms.ollama import Ollama

from genai_template.config.ollama import resolve_ollama_base_url
from genai_template.workflow.portfolio.adapters.generation.base import (
    LlamaIndexStructuredSummaryGenerator,
    optional_non_negative_int,
    optional_string,
    read_value,
)
from genai_template.workflow.portfolio.domain.generation import (
    ProviderAuditMetadata,
    TokenUsage,
)


class OllamaStructuredSummaryGenerator(LlamaIndexStructuredSummaryGenerator[Any]):
    """Generate validated summaries with a configured Ollama model.

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
    ) -> None:
        """Initialize the Ollama structured generation adapter.

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
                Optional Ollama endpoint override.
        """

        options: dict[str, object] = {}
        if seed is not None:
            options["seed"] = seed
        llm = Ollama(
            model=model,
            base_url=base_url or resolve_ollama_base_url(),
            temperature=temperature,
            request_timeout=timeout_seconds,
            additional_kwargs=options,
            json_mode=True,
        )
        super().__init__(llm=llm, model=model)

    def _extract_accounting(
        self, completion: Any
    ) -> tuple[TokenUsage, ProviderAuditMetadata]:
        """Extract Ollama token usage and safe response metadata.

        Args:
            completion:
                LlamaIndex Ollama completion response.

        Returns:
            Token usage and safe audit metadata.
        """

        raw = completion.raw
        return (
            TokenUsage(
                input_tokens=optional_non_negative_int(
                    read_value(raw, "prompt_eval_count")
                ),
                output_tokens=optional_non_negative_int(read_value(raw, "eval_count")),
            ),
            ProviderAuditMetadata(
                finish_reason=optional_string(read_value(raw, "done_reason"))
            ),
        )
