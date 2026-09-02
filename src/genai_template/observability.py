"""OpenInference-compatible spans for the RAG answer path."""

from __future__ import annotations

import json
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span

OPENINFERENCE_SPAN_KIND = "openinference.span.kind"
INPUT_VALUE = "input.value"
OUTPUT_VALUE = "output.value"
RETRIEVAL_DOCUMENTS = "retrieval.documents"


@contextmanager
def application_span(
    name: str,
    kind: str,
    attributes: Mapping[str, Any] | None = None,
) -> Generator[Span, None, None]:
    """Create an application span when a configured provider is active.

    The OpenTelemetry proxy provider makes this a no-op when Phoenix tracing is
    disabled, so the answer path does not need a configuration branch.
    """

    tracer = trace.get_tracer("genai_template.rag")
    with tracer.start_as_current_span(name) as span:
        span.set_attribute(OPENINFERENCE_SPAN_KIND, kind)
        for key, value in (attributes or {}).items():
            if value is not None:
                span.set_attribute(key, value)
        yield span


def retrieved_documents_attribute(documents: list[dict[str, Any]]) -> str:
    """Serialize retrieved chunks for the OpenInference retrieval attribute."""

    return json.dumps(documents, default=str)
