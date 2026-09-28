"""Opt-in integration smoke tests for plain-text generation providers."""

from __future__ import annotations

import os
from typing import Literal

import pytest

from genai_template.workflow.portfolio.adapters.generators import (
    OllamaTextGenerator,
    OpenAITextGenerator,
)
from genai_template.workflow.portfolio.domain import (
    ArtifactProvenance,
    GenerationRequest,
)


def _request(provider: Literal["ollama", "openai"], model: str) -> GenerationRequest:
    """Build a minimal provider smoke-test request.

    Args:
        provider:
            Provider identity.
        model:
            Provider model name.

    Returns:
        Fully identified component request.
    """

    return GenerationRequest(
        provenance=ArtifactProvenance(
            artifact_kind="component",
            generation_fingerprint="1" * 64,
            source_fingerprint="2" * 64,
            unit_input_fingerprint="3" * 64,
            prompt_id="portfolio-component-v1",
            prompt_hash="4" * 64,
            output_schema_version="v1",
            provider=provider,
            model=model,
            temperature=0,
        ),
        prompt="Return one plain Markdown sentence about a function named run.",
        input_paths=("src/a.py",),
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or not os.getenv("PORTFOLIO_OPENAI_MODEL"),
    reason="OPENAI_API_KEY and PORTFOLIO_OPENAI_MODEL are required.",
)
def test_openai_text_generation_smoke() -> None:
    """OpenAI should return non-empty plain text."""

    model = os.environ["PORTFOLIO_OPENAI_MODEL"]
    generator = OpenAITextGenerator(model, temperature=0, timeout_seconds=180)

    response = generator.generate(_request("openai", model))

    assert response.text.strip()
    assert response.provider == "openai"


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("PORTFOLIO_OLLAMA_MODEL"),
    reason="PORTFOLIO_OLLAMA_MODEL is required for the Ollama smoke test.",
)
def test_ollama_text_generation_smoke() -> None:
    """Ollama should return non-empty plain text."""

    model = os.environ["PORTFOLIO_OLLAMA_MODEL"]
    generator = OllamaTextGenerator(
        model,
        temperature=0,
        seed=7,
        timeout_seconds=180,
    )

    response = generator.generate(_request("ollama", model))

    assert response.text.strip()
    assert response.provider == "ollama"
