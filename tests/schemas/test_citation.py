"""Tests for citation provenance schemas."""

from genai_template.schemas import CitationSource


def test_generic_citation_omits_optional_portfolio_provenance() -> None:
    """Generic citation serialization should retain its existing shape."""

    source = CitationSource(
        label="S1",
        chunk_id="guide.md-000",
        document_name="guide.md",
        section=None,
        content="Generic content",
        distance=0.2,
        cited=False,
    )

    assert source.project_slug is None
    assert source.model_dump() == {
        "label": "S1",
        "chunk_id": "guide.md-000",
        "document_name": "guide.md",
        "section": None,
        "content": "Generic content",
        "distance": 0.2,
        "cited": False,
    }


def test_portfolio_citation_serializes_complete_provenance() -> None:
    """Portfolio citation serialization should expose every audit identity."""

    source = CitationSource(
        label="S1",
        chunk_id="alpha--overview.md-000",
        document_name="alpha--overview.md",
        section="/Overview/",
        content="Portfolio content",
        distance=0.1,
        cited=True,
        project_slug="alpha",
        project_display_name="Alpha",
        document_type="overview",
        repository_url="https://example.com/alpha.git",
        resolved_commit_sha="a" * 40,
        corpus_fingerprint="b" * 64,
        generation_fingerprint="c" * 64,
    )

    assert source.model_dump() == {
        "label": "S1",
        "chunk_id": "alpha--overview.md-000",
        "document_name": "alpha--overview.md",
        "section": "/Overview/",
        "content": "Portfolio content",
        "distance": 0.1,
        "cited": True,
        "project_slug": "alpha",
        "project_display_name": "Alpha",
        "document_type": "overview",
        "repository_url": "https://example.com/alpha.git",
        "resolved_commit_sha": "a" * 40,
        "corpus_fingerprint": "b" * 64,
        "generation_fingerprint": "c" * 64,
    }
