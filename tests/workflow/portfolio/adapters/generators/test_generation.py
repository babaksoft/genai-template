"""Tests for official-SDK plain-text generation adapters."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import patch

import httpx
import pytest
from ollama import ChatResponse, Message
from openai import APITimeoutError

from genai_template.workflow.portfolio.config.models import TokenPricingConfig
from genai_template.workflow.portfolio.domain import (
    ArtifactProvenance,
    GenerationRequest,
    TextGenerationError,
    TokenUsage,
)
from genai_template.workflow.portfolio.infrastructure.generation import (
    estimate_generation_cost,
)
from genai_template.workflow.portfolio.infrastructure.generators import (
    OllamaTextGenerator,
    OpenAITextGenerator,
)

_TEXT = "## Responsibilities\n\nFact.\n\n### Evidence\n\n- src/a.py"


class _Responses:
    """Record one fake Responses API call.

    Attributes:
        response:
            Response value or exception supplied by the test.
        calls:
            Keyword arguments received by the SDK boundary.
    """

    def __init__(self, response: object) -> None:
        """Initialize deterministic response behavior.

        Args:
            response:
                Value to return or exception to raise.
        """

        self.response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        """Record and return one configured response.

        Args:
            **kwargs:
                Responses API request arguments.

        Returns:
            Configured fake response.

        Raises:
            Exception:
                Configured fake provider exception.
        """

        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class _OpenAIClient:
    """Minimal injected OpenAI client exposing the Responses resource."""

    def __init__(self, response: object) -> None:
        """Initialize the fake resource.

        Args:
            response:
                Value or exception supplied by the test.
        """

        self.responses = _Responses(response)


class _OllamaClient:
    """Record one fake Ollama chat call.

    Attributes:
        response:
            Chat response or exception supplied by the test.
        calls:
            Keyword arguments received by the SDK boundary.
    """

    def __init__(self, response: object) -> None:
        """Initialize deterministic chat behavior.

        Args:
            response:
                Value to return or exception to raise.
        """

        self.response = response
        self.calls: list[dict[str, object]] = []

    def chat(self, **kwargs: object) -> object:
        """Record and return one configured response.

        Args:
            **kwargs:
                Ollama chat request arguments.

        Returns:
            Configured fake response.

        Raises:
            Exception:
                Configured fake provider exception.
        """

        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _request(
    provider: Literal["ollama", "openai"], model: str = "test-model"
) -> GenerationRequest:
    """Build a valid component text-generation request.

    Args:
        provider:
            Provider identity for request provenance.
        model:
            Model identity for request provenance.

    Returns:
        Valid generation request.
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
        prompt="Safe assembled Markdown prompt",
        input_paths=("src/a.py",),
    )


def _openai_response(
    *,
    text: str = _TEXT,
    status: str = "completed",
    model: str = "test-model",
    refusal: bool = False,
    usage: object | None = None,
) -> Any:
    """Build the response shape consumed from the OpenAI SDK.

    Args:
        text:
            Aggregated response text.
        status:
            Responses API completion status.
        model:
            Returned model identity.
        refusal:
            Whether output contains an explicit refusal item.
        usage:
            Optional provider usage value.

    Returns:
        SDK-shaped fake response.
    """

    content_type = "refusal" if refusal else "output_text"
    return SimpleNamespace(
        id="response-1",
        model=model,
        status=status,
        output_text=text,
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type=content_type)],
            )
        ],
        usage=usage,
    )


def _ollama_response(
    *,
    text: str = _TEXT,
    done: bool = True,
    done_reason: str = "stop",
    model: str = "test-model",
    input_tokens: int | None = 9,
    output_tokens: int | None = 4,
) -> ChatResponse:
    """Build an official Ollama SDK chat response.

    Args:
        text:
            Assistant response content.
        done:
            Provider completion indicator.
        done_reason:
            Provider completion reason.
        model:
            Returned model identity.
        input_tokens:
            Optional input-token count.
        output_tokens:
            Optional output-token count.

    Returns:
        Typed Ollama response.
    """

    return ChatResponse(
        model=model,
        done=done,
        done_reason=done_reason,
        prompt_eval_count=input_tokens,
        eval_count=output_tokens,
        message=Message(role="assistant", content=text),
    )


def test_openai_returns_plain_text_usage_and_makes_one_request() -> None:
    """OpenAI uses one plain Responses call without format or repair settings."""

    usage = SimpleNamespace(input_tokens=12, output_tokens=7)
    client = _OpenAIClient(_openai_response(usage=usage))
    generator = OpenAITextGenerator(
        "test-model", temperature=0.2, timeout_seconds=5, client=client  # type: ignore[arg-type]
    )

    response = generator.generate(_request("openai"))

    assert response.text == _TEXT
    assert response.token_usage == TokenUsage(input_tokens=12, output_tokens=7)
    assert response.provider_metadata.request_id == "response-1"
    assert client.responses.calls == [
        {
            "model": "test-model",
            "input": "Safe assembled Markdown prompt",
            "temperature": 0.2,
            "store": False,
        }
    ]
    assert "text" not in client.responses.calls[0]


def test_openai_sdk_automatic_retries_are_disabled() -> None:
    """The production OpenAI client is constructed with zero SDK retries."""

    with patch(
        "genai_template.workflow.portfolio.adapters.generators.openai.OpenAI"
    ) as client_class:
        OpenAITextGenerator("test-model", temperature=0, timeout_seconds=12)

    client_class.assert_called_once_with(timeout=12, max_retries=0)


def test_ollama_returns_plain_text_usage_and_makes_one_request() -> None:
    """Ollama uses one plain chat call without JSON format or retries."""

    client = _OllamaClient(_ollama_response())
    generator = OllamaTextGenerator(
        "test-model",
        temperature=0.1,
        seed=7,
        timeout_seconds=5,
        client=client,  # type: ignore[arg-type]
    )

    response = generator.generate(_request("ollama"))

    assert response.text == _TEXT
    assert response.token_usage == TokenUsage(input_tokens=9, output_tokens=4)
    assert client.calls == [
        {
            "model": "test-model",
            "messages": (
                {"role": "user", "content": "Safe assembled Markdown prompt"},
            ),
            "stream": False,
            "options": {"temperature": 0.1, "seed": 7},
        }
    ]
    assert "format" not in client.calls[0]


def test_ollama_accepts_exact_cloud_response_model_alias() -> None:
    """Ollama may omit the Cloud routing suffix from response attribution."""

    client = _OllamaClient(_ollama_response(model="test-model"))
    generator = OllamaTextGenerator(
        "test-model-cloud",
        temperature=0,
        seed=7,
        timeout_seconds=5,
        client=client,  # type: ignore[arg-type]
    )

    response = generator.generate(_request("ollama", "test-model-cloud"))

    assert response.model == "test-model-cloud"
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "generator",
    [
        OpenAITextGenerator(
            "test-model",
            temperature=0,
            timeout_seconds=5,
            client=_OpenAIClient(_openai_response(usage=None)),  # type: ignore[arg-type]
        ),
        OllamaTextGenerator(
            "test-model",
            temperature=0,
            seed=None,
            timeout_seconds=5,
            client=_OllamaClient(
                _ollama_response(input_tokens=None, output_tokens=None)
            ),  # type: ignore[arg-type]
        ),
    ],
)
def test_unavailable_usage_remains_unknown(generator: Any) -> None:
    """Either provider may omit token counts without inventing zero values.

    Args:
        generator:
            Configured plain-text adapter with absent usage.
    """

    provider = generator.provider
    response = generator.generate(_request(provider))

    assert response.token_usage == TokenUsage()


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (_openai_response(refusal=True), "refusal"),
        (_openai_response(status="incomplete"), "incomplete"),
        (_openai_response(text="  "), "empty-output"),
        (_openai_response(model="other-model"), "identity-mismatch"),
        (RuntimeError("secret provider detail"), "provider-failure"),
        (
            APITimeoutError(request=httpx.Request("POST", "https://example.test")),
            "provider-timeout",
        ),
    ],
)
def test_openai_normalizes_failure_modes(response: object, reason: str) -> None:
    """OpenAI failures have stable reasons and do not leak provider content.

    Args:
        response:
            Fake SDK response or exception.
        reason:
            Expected stable failure reason.
    """

    client = _OpenAIClient(response)
    generator = OpenAITextGenerator(
        "test-model", temperature=0, timeout_seconds=5, client=client  # type: ignore[arg-type]
    )

    with pytest.raises(TextGenerationError) as caught:
        generator.generate(_request("openai"))

    assert caught.value.reason == reason
    assert "secret provider detail" not in str(caught.value)
    assert len(client.responses.calls) == 1


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (_ollama_response(done_reason="refusal"), "refusal"),
        (_ollama_response(done=False), "incomplete"),
        (_ollama_response(done_reason="length"), "incomplete"),
        (_ollama_response(text=""), "empty-output"),
        (_ollama_response(model="other-model"), "identity-mismatch"),
        (RuntimeError("secret provider detail"), "provider-failure"),
        (TimeoutError("secret timeout detail"), "provider-timeout"),
    ],
)
def test_ollama_normalizes_failure_modes(response: object, reason: str) -> None:
    """Ollama failures have stable reasons and do not leak provider content.

    Args:
        response:
            Fake SDK response or exception.
        reason:
            Expected stable failure reason.
    """

    client = _OllamaClient(response)
    generator = OllamaTextGenerator(
        "test-model",
        temperature=0,
        seed=None,
        timeout_seconds=5,
        client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(TextGenerationError) as caught:
        generator.generate(_request("ollama"))

    assert caught.value.reason == reason
    assert "secret" not in str(caught.value)
    assert len(client.calls) == 1


def test_request_identity_mismatch_makes_no_provider_request() -> None:
    """Mismatched provenance fails before any SDK invocation."""

    client = _OllamaClient(_ollama_response())
    generator = OllamaTextGenerator(
        "test-model",
        temperature=0,
        seed=None,
        timeout_seconds=5,
        client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(TextGenerationError) as caught:
        generator.generate(_request("openai"))

    assert caught.value.reason == "identity-mismatch"
    assert client.calls == []


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
