"""Tests for structured generator boundaries and generation accounting."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from genai_template.workflow.portfolio.adapters.generators.base import (
    LlamaIndexStructuredSummaryGenerator,
)
from genai_template.workflow.portfolio.config.models import TokenPricingConfig
from genai_template.workflow.portfolio.domain import (
    ArtifactProvenance,
    ComponentSummary,
    GenerationRequest,
    StructuredGenerationError,
    TokenUsage,
)
from genai_template.workflow.portfolio.generation import estimate_generation_cost

_VALID_OUTPUT = """{
  "responsibilities": {"content": ["Fact."], "evidence_paths": ["src/a.py"]},
  "important_abstractions": {"content": ["Fact."], "evidence_paths": ["src/a.py"]},
  "behavior": {"content": ["Fact."], "evidence_paths": ["src/a.py"]},
  "constraints": {"content": ["Fact."], "evidence_paths": ["src/a.py"]},
  "testing_evidence": {"content": ["Fact."], "evidence_paths": ["src/a.py"]}
}"""


class _Completion:
    """Minimal deterministic LlamaIndex-like completion response.

    Attributes:
        text:
            Completion text returned by the fake provider.
        raw:
            Absent raw provider response.
    """

    def __init__(self, text: str) -> None:
        """Initialize the completion.

        Args:
            text:
                Completion text.
        """

        self.text = text
        self.raw = None


class _FakeLlm:
    """Deterministic completion provider used without network access.

    Attributes:
        text:
            Completion text returned by the fake.
        error:
            Optional configured provider failure.
        last_prompt:
            Most recent prompt passed to the fake.
    """

    def __init__(
        self, text: str = _VALID_OUTPUT, error: Exception | None = None
    ) -> None:
        """Initialize provider behavior.

        Args:
            text:
                Completion text to return.
            error:
                Optional provider error to raise.
        """

        self.text = text
        self.error = error
        self.last_prompt: str | None = None

    def complete(self, prompt: str) -> _Completion:
        """Return a deterministic completion.

        Args:
            prompt:
                Schema-formatted provider prompt.

        Returns:
            Deterministic completion.

        Raises:
            Exception:
                Configured fake failure.
        """

        self.last_prompt = prompt
        if self.error is not None:
            raise self.error
        return _Completion(self.text)


class _FakeGenerator(LlamaIndexStructuredSummaryGenerator[Any]):
    """Concrete deterministic generator for adapter contract tests.

    Attributes:
        provider:
            Fake provider identity matching test provenance.
    """

    provider = "ollama"


def _request() -> GenerationRequest:
    """Build a valid component generation request.

    Returns:
        Valid generation request.
    """

    provenance = ArtifactProvenance(
        artifact_kind="component",
        generation_fingerprint="1" * 64,
        source_fingerprint="2" * 64,
        unit_input_fingerprint="3" * 64,
        prompt_id="portfolio-component-v1",
        prompt_hash="4" * 64,
        output_schema_version="v1",
        provider="ollama",
        model="test-model",
        temperature=0,
    )
    return GenerationRequest(
        provenance=provenance,
        prompt="Safe assembled prompt",
        input_paths=("src/a.py",),
    )


def test_fake_generator_returns_typed_output_and_unknown_usage() -> None:
    """Validated JSON becomes a typed value and absent usage stays unknown."""

    llm = _FakeLlm()
    generator = _FakeGenerator(llm=llm, model="test-model")

    response = generator.generate(_request(), ComponentSummary)

    assert isinstance(response.value, ComponentSummary)
    assert response.token_usage == TokenUsage()
    assert llm.last_prompt is not None
    assert "JSON schema" in llm.last_prompt


@pytest.mark.parametrize(
    "llm, reason",
    [
        (_FakeLlm(text="not json"), "invalid-output"),
        (_FakeLlm(error=RuntimeError("secret provider detail")), "provider-failure"),
    ],
)
def test_generator_normalizes_validation_and_provider_failures(
    llm: _FakeLlm, reason: str
) -> None:
    """Raw failures do not leak response or provider content in messages.

    Args:
        llm:
            Configured deterministic provider.
        reason:
            Expected normalized failure reason.
    """

    generator = _FakeGenerator(llm=llm, model="test-model")

    with pytest.raises(StructuredGenerationError) as caught:
        generator.generate(_request(), ComponentSummary)

    assert caught.value.reason == reason
    assert "secret provider detail" not in str(caught.value)


def test_cost_requires_complete_usage_and_uses_explicit_rates() -> None:
    """Complete, partial, and unavailable accounting remain distinguishable."""

    pricing = TokenPricingConfig(
        input_per_million_tokens=Decimal(2),
        output_per_million_tokens=Decimal(8),
    )

    assert estimate_generation_cost(
        TokenUsage(input_tokens=100, output_tokens=25), pricing
    ) == Decimal("0.0004")
    assert estimate_generation_cost(TokenUsage(input_tokens=100), pricing) is None
    assert estimate_generation_cost(TokenUsage(), pricing) is None
