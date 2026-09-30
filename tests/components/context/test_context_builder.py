"""Tests for the ContextBuilder."""

from typing import Any

import pytest

from genai_template.components.context import ContextBuilder
from genai_template.components.readers.portfolio_metadata import (
    COMPONENT_ID,
    CORPUS_FINGERPRINT,
    DOCUMENT_TYPE,
    FILE_NAME,
    GENERATION_FINGERPRINT,
    HEADER_PATH,
    PROJECT_DISPLAY_NAME,
    PROJECT_SLUG,
    REPOSITORY_URL,
    RESOLVED_COMMIT_SHA,
)
from genai_template.schemas import DocumentChunk, RetrievedChunk


def create_retrieved_chunk(
    text: str,
    *,
    chunk_id: str = "chunk-id",
    document_id: str = "document-id",
    metadata: dict[str, Any] | None = None,
    distance: float = 0.0,
) -> RetrievedChunk:
    """Create a retrieved chunk for testing.

    Args:
        text:
            Chunk content.
        chunk_id:
            Chunk identifier.
        document_id:
            Parent document identifier.
        metadata:
            Optional chunk metadata.
        distance:
            Retrieval distance.

    Returns:
        Retrieved chunk fixture.
    """

    return RetrievedChunk(
        chunk=DocumentChunk(
            id=chunk_id,
            document_id=document_id,
            text=text,
            metadata=metadata or {},
            embedding=None,
        ),
        distance=distance,
    )


def portfolio_metadata(
    project_slug: str,
    display_name: str,
    *,
    document_type: str = "overview",
    component_id: str | None = None,
    repository_url: str | None = "https://example.com/repository.git",
    commit_character: str = "a",
) -> dict[str, Any]:
    """Create valid retrieved Portfolio metadata.

    Args:
        project_slug:
            Stable project identifier.
        display_name:
            Human-readable project name.
        document_type:
            Generated document type.
        component_id:
            Optional component identifier.
        repository_url:
            Optional normalized repository URL.
        commit_character:
            Character used for the test commit hash.

    Returns:
        Complete controlled Portfolio metadata.
    """

    metadata = {
        PROJECT_SLUG: project_slug,
        PROJECT_DISPLAY_NAME: display_name,
        DOCUMENT_TYPE: document_type,
        RESOLVED_COMMIT_SHA: commit_character * 40,
        CORPUS_FINGERPRINT: "c" * 64,
        GENERATION_FINGERPRINT: "d" * 64,
        FILE_NAME: f"{project_slug}--{document_type}.md",
        HEADER_PATH: "/Shared/",
    }
    if component_id is not None:
        metadata[COMPONENT_ID] = component_id
        metadata[FILE_NAME] = f"{project_slug}--component--{component_id}.md"
    if repository_url is not None:
        metadata[REPOSITORY_URL] = repository_url
    return metadata


def test_build_returns_empty_context_for_empty_input() -> None:
    """An empty retrieval result should produce an empty citation context."""

    context = ContextBuilder().build([])

    assert context.text == ""
    assert context.sources == []


def test_build_formats_source_metadata_and_markdown_section() -> None:
    """Context blocks should include safe document and section metadata."""

    context = ContextBuilder().build(
        [
            create_retrieved_chunk(
                "Important content",
                metadata={
                    "file_name": "/private/corpus/guide.md",
                    "file_path": "/private/corpus/guide.md",
                    "header_path": "/Introduction/Overview/",
                },
                distance=0.123,
            )
        ]
    )

    assert context.text == (
        "[S1]\n"
        "Document: guide.md\n"
        "Section: /Introduction/Overview/\n"
        "Content:\n"
        "Important content"
    )
    assert context.sources[0].model_dump() == {
        "label": "S1",
        "chunk_id": "chunk-id",
        "document_name": "guide.md",
        "section": "/Introduction/Overview/",
        "content": "Important content",
        "distance": 0.123,
        "cited": False,
    }
    assert "/private/corpus" not in context.text


def test_build_preserves_order_and_labels_duplicate_documents() -> None:
    """Each retrieved chunk should keep its order and receive a distinct label."""

    context = ContextBuilder().build(
        [
            create_retrieved_chunk(
                "First", chunk_id="one", metadata={"file_name": "guide.md"}
            ),
            create_retrieved_chunk(
                "Second", chunk_id="two", metadata={"file_name": "guide.md"}
            ),
            create_retrieved_chunk(
                "Third", chunk_id="three", metadata={"file_name": "other.md"}
            ),
        ]
    )

    assert [source.label for source in context.sources] == ["S1", "S2", "S3"]
    assert [source.chunk_id for source in context.sources] == ["one", "two", "three"]
    assert [source.document_name for source in context.sources] == [
        "guide.md",
        "guide.md",
        "other.md",
    ]
    assert context.text.index("First") < context.text.index("Second")
    assert context.text.index("Second") < context.text.index("Third")


def test_build_falls_back_to_safe_document_id_basename() -> None:
    """Missing file-name metadata should safely fall back to document ID."""

    context = ContextBuilder().build(
        [
            create_retrieved_chunk("Unix", document_id="/private/docs/unix.md"),
            create_retrieved_chunk(
                "Windows", document_id=r"C:\private\docs\windows.md"
            ),
        ]
    )

    assert [source.document_name for source in context.sources] == [
        "unix.md",
        "windows.md",
    ]
    assert all(source.section is None for source in context.sources)
    assert "/private/docs" not in context.text
    assert r"C:\private\docs" not in context.text


def test_build_disambiguates_projects_with_shared_sections() -> None:
    """Portfolio context should distinguish projects and repository revisions."""

    context = ContextBuilder().build(
        [
            create_retrieved_chunk(
                "Alpha details",
                chunk_id="alpha--overview.md-000",
                document_id="alpha--overview.md",
                metadata=portfolio_metadata("alpha", "Alpha", commit_character="a"),
            ),
            create_retrieved_chunk(
                "Beta details",
                chunk_id="beta--overview.md-000",
                document_id="beta--overview.md",
                metadata=portfolio_metadata("beta", "Beta", commit_character="b"),
            ),
        ]
    )

    assert "Project: Alpha (alpha)" in context.text
    assert "Project: Beta (beta)" in context.text
    assert f"Revision: {'a' * 40}" in context.text
    assert f"Revision: {'b' * 40}" in context.text
    assert context.text.count("Section: /Shared/") == 2
    assert [source.project_slug for source in context.sources] == ["alpha", "beta"]
    assert [source.resolved_commit_sha for source in context.sources] == [
        "a" * 40,
        "b" * 40,
    ]


def test_build_formats_component_provenance_without_absent_repository() -> None:
    """Component context should omit an unconfigured repository URL."""

    metadata = portfolio_metadata(
        "alpha",
        "Alpha",
        document_type="component",
        component_id="api",
        repository_url=None,
    )
    context = ContextBuilder().build(
        [
            create_retrieved_chunk(
                "API details",
                document_id="alpha--component--api.md",
                metadata=metadata,
            )
        ]
    )

    assert "Document type: component" in context.text
    assert "Component: api" in context.text
    assert "Repository:" not in context.text
    assert "None" not in context.text
    assert context.sources[0].component_id == "api"
    assert context.sources[0].repository_url is None


@pytest.mark.parametrize(
    "metadata",
    [
        {PROJECT_SLUG: "alpha", FILE_NAME: "alpha--overview.md"},
        {
            **portfolio_metadata("alpha", "Alpha"),
            PROJECT_DISPLAY_NAME: 42,
        },
        {
            **portfolio_metadata("alpha", "Alpha"),
            "file_path": "/private/releases/release-1/alpha--overview.md",
        },
    ],
)
def test_build_rejects_partial_or_malformed_portfolio_metadata(
    metadata: dict[str, Any],
) -> None:
    """Claimed Portfolio provenance must satisfy the full controlled contract.

    Args:
        metadata:
            Invalid metadata under test.
    """

    with pytest.raises(ValueError, match="Portfolio metadata"):
        ContextBuilder().build([create_retrieved_chunk("Unsafe", metadata=metadata)])


def test_build_keeps_generic_metadata_compatible() -> None:
    """Generic chunks should not receive invented Portfolio provenance."""

    context = ContextBuilder().build(
        [
            create_retrieved_chunk(
                "Generic",
                metadata={
                    FILE_NAME: "/private/source/guide.md",
                    HEADER_PATH: "/Guide/",
                    "file_path": "/private/source/guide.md",
                },
            )
        ]
    )

    assert "Project:" not in context.text
    assert "/private/source" not in context.text
    assert context.sources[0].project_slug is None
    assert context.sources[0].model_dump() == {
        "label": "S1",
        "chunk_id": "chunk-id",
        "document_name": "guide.md",
        "section": "/Guide/",
        "content": "Generic",
        "distance": 0.0,
        "cited": False,
    }
