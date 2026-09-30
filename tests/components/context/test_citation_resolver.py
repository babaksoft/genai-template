"""Tests for generated-answer citation resolution."""

import pytest

from genai_template.components.context import resolve_citations
from genai_template.schemas import CitationSource


def create_sources(count: int = 3) -> list[CitationSource]:
    """Create ordered citation sources.

    Args:
        count:
            Number of sources to create.

    Returns:
        Citation source fixtures.
    """

    return [
        CitationSource(
            label=f"S{index}",
            chunk_id=f"chunk-{index}",
            document_name="guide.md",
            section=None,
            content=f"Content {index}",
            distance=float(index),
            cited=False,
        )
        for index in range(1, count + 1)
    ]


@pytest.mark.parametrize(
    ("answer", "expected_flags"),
    [
        ("Supported [S1].", [True, False, False]),
        ("Repeated [S2] and again [S2].", [False, True, False]),
        ("Out of order [S3], then [S1].", [True, False, True]),
        ("No citations.", [False, False, False]),
        ("Ignored [S0], [S-1], and [source].", [False, False, False]),
    ],
)
def test_resolve_citations_marks_only_supported_labels(
    answer: str, expected_flags: list[bool]
) -> None:
    """Only recognized labels supplied in context should mark a source cited.

    Args:
        answer:
            Generated answer under test.
        expected_flags:
            Expected cited flags in source order.
    """

    sources, warnings = resolve_citations(answer, create_sources())

    assert [source.cited for source in sources] == expected_flags
    assert warnings == []


def test_resolve_citations_warns_once_in_unique_numeric_order() -> None:
    """Unsupported labels should produce one deterministic non-fatal warning."""

    answer = "Claims [S20], [S4], [S20], and supported [S2]."

    sources, warnings = resolve_citations(answer, create_sources())

    assert [source.cited for source in sources] == [False, True, False]
    assert [warning.model_dump() for warning in warnings] == [
        {
            "code": "unsupported_citation_labels",
            "message": "The answer references labels not present in its context.",
            "labels": ["S4", "S20"],
        }
    ]


def test_resolve_citations_preserves_portfolio_provenance() -> None:
    """Citation resolution should change only the request-local cited flag."""

    source = CitationSource(
        label="S1",
        chunk_id="alpha--component--api.md-000",
        document_name="alpha--component--api.md",
        section="/API/",
        content="API content",
        distance=0.1,
        cited=False,
        project_slug="alpha",
        project_display_name="Alpha",
        document_type="component",
        component_id="api",
        repository_url="https://example.com/alpha.git",
        resolved_commit_sha="a" * 40,
        corpus_fingerprint="b" * 64,
        generation_fingerprint="c" * 64,
    )

    resolved, warnings = resolve_citations("Supported [S1].", [source])

    assert warnings == []
    assert resolved[0].model_dump(exclude={"cited"}) == source.model_dump(
        exclude={"cited"}
    )
    assert resolved[0].cited is True
