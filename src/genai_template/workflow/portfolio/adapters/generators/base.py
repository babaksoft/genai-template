"""Shared plain-text provider adapter behavior."""

from __future__ import annotations

from genai_template.workflow.portfolio.domain.errors import TextGenerationError
from genai_template.workflow.portfolio.domain.generation import GenerationRequest


class TextGeneratorBase:
    """Validate common request and response invariants for text adapters.

    Attributes:
        provider:
            Stable provider identifier supplied by concrete adapters.
    """

    provider: str

    def __init__(self, *, model: str) -> None:
        """Initialize the shared adapter identity.

        Args:
            model:
                Exact configured provider model.
        """

        self._model = model

    def _validate_request_identity(self, request: GenerationRequest) -> None:
        """Ensure request provenance agrees with the concrete adapter.

        Args:
            request:
                Request whose provider identity is checked.

        Raises:
            TextGenerationError:
                If provider or model provenance does not match this adapter.
        """

        provenance = request.provenance
        if provenance.provider != self.provider or provenance.model != self._model:
            self._raise(
                "generation request provider identity does not match adapter",
                "identity-mismatch",
            )

    def _validate_response_model(self, model: object) -> None:
        """Reject a provider response attributed to another model.

        Args:
            model:
                Provider-returned model identity.

        Raises:
            TextGenerationError:
                If the identity is absent or differs from configuration.
        """

        if model != self._model:
            self._raise(
                "generation response model identity does not match request",
                "identity-mismatch",
            )

    def _raise(self, message: str, reason: str) -> None:
        """Raise one content-safe provider-neutral failure.

        Args:
            message:
                Safe human-readable failure description.
            reason:
                Stable machine-readable failure reason.

        Raises:
            TextGenerationError:
                Always.
        """

        raise TextGenerationError(
            message,
            provider=self.provider,
            model=self._model,
            reason=reason,
        )


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
