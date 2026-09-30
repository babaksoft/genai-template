"""Tests for manifest-aware Portfolio document loading."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from llama_index.core import Document

from genai_template.components.readers import PortfolioReader
from genai_template.components.readers.portfolio_metadata import (
    COMPONENT_ID,
    CORPUS_FINGERPRINT,
    DOCUMENT_TYPE,
    FILE_NAME,
    GENERATION_FINGERPRINT,
    PROJECT_DISPLAY_NAME,
    PROJECT_SLUG,
    REPOSITORY_URL,
    RESOLVED_COMMIT_SHA,
)
from genai_template.components.splitters import (
    DocumentSplitter,
    MarkdownDocumentSplitter,
)
from genai_template.schemas import LoadedDocuments
from genai_template.workflow.portfolio.corpus import (
    CorpusValidationError,
    build_corpus_manifest_v2,
    manifest_bytes,
)
from genai_template.workflow.portfolio.domain import (
    ArtifactKind,
    ManifestDocumentV2,
    ManifestProject,
    ManifestPrompt,
)


def _write_corpus(directory: Path) -> tuple[str, dict[str, dict[str, str]]]:
    """Write a valid two-project Portfolio corpus fixture.

    Args:
        directory:
            Release directory to populate.

    Returns:
        Corpus fingerprint and expected metadata keyed by filename.
    """

    specifications: tuple[tuple[str, str, str, ArtifactKind, str | None, str], ...] = (
        ("alpha", "Alpha", "alpha--overview.md", "overview", None, "a"),
        ("beta", "Beta", "beta--component--api.md", "component", "api", "b"),
    )
    projects = tuple(
        ManifestProject(
            project_slug=slug,
            project_display_name=display_name,
            repository_url=f"https://example.test/{slug}.git",
            requested_ref="main",
            resolved_commit_sha=character * 40,
            source_fingerprint=character * 64,
            generation_profile="balanced",
            prompt_version="v1",
            prompts=(ManifestPrompt(prompt_id="test.v1", prompt_hash="c" * 64),),
            output_schema_version="v1",
            provider="openai",
            model=f"model-{slug}",
            configuration_fingerprint="d" * 64,
        )
        for slug, display_name, _, _, _, character in specifications
    )
    contents = {
        filename: f"# {display_name}\n\nPortable content for {slug}.\n"
        for slug, display_name, filename, _, _, _ in specifications
    }
    records = tuple(
        ManifestDocumentV2(
            project_slug=slug,
            filename=filename,
            document_type=document_type,
            component_id=component_id,
            evidence_paths=("README.md",),
            generation_fingerprint=character * 64,
            artifact_hash="e" * 64,
            content_hash=hashlib.sha256(contents[filename].encode()).hexdigest(),
            byte_size=len(contents[filename].encode()),
        )
        for slug, _, filename, document_type, component_id, character in specifications
    )
    manifest = build_corpus_manifest_v2(projects=projects, documents=records)
    directory.mkdir()
    for filename, content in contents.items():
        (directory / filename).write_text(content, encoding="utf-8")
    (directory / "manifest.json").write_bytes(manifest_bytes(manifest))

    project_by_slug = {project.project_slug: project for project in projects}
    expected: dict[str, dict[str, str]] = {}
    for record in records:
        project = project_by_slug[record.project_slug]
        metadata = {
            PROJECT_SLUG: project.project_slug,
            PROJECT_DISPLAY_NAME: project.project_display_name,
            DOCUMENT_TYPE: record.document_type,
            REPOSITORY_URL: str(project.repository_url),
            RESOLVED_COMMIT_SHA: project.resolved_commit_sha,
            CORPUS_FINGERPRINT: manifest.corpus_fingerprint,
            GENERATION_FINGERPRINT: record.generation_fingerprint,
            FILE_NAME: record.filename,
        }
        if record.component_id is not None:
            metadata[COMPONENT_ID] = record.component_id
        expected[record.filename] = metadata
    return manifest.corpus_fingerprint, expected


def test_load_projects_controlled_metadata_and_portable_ids(tmp_path: Path) -> None:
    """Portfolio documents should contain only manifest-backed metadata."""

    release = tmp_path / "release"
    fingerprint, expected = _write_corpus(release)

    loaded = PortfolioReader().load(release)

    assert isinstance(loaded, LoadedDocuments)
    assert loaded.provenance is not None
    assert loaded.provenance.corpus_fingerprint == fingerprint
    assert [document.id_ for document in loaded] == sorted(expected)
    assert {document.id_: document.metadata for document in loaded} == expected
    assert str(release) not in repr([document.metadata for document in loaded])


@pytest.mark.parametrize("splitter", [DocumentSplitter(), MarkdownDocumentSplitter()])
def test_splitters_preserve_provenance_and_stable_ids(
    tmp_path: Path,
    splitter: DocumentSplitter | MarkdownDocumentSplitter,
) -> None:
    """Both splitters should retain provenance under deterministic chunk IDs."""

    release = tmp_path / "release"
    _, expected = _write_corpus(release)
    loaded = PortfolioReader().load(release)

    chunks = splitter.split(list(loaded.documents))

    assert chunks
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    for chunk in chunks:
        assert chunk.id.startswith(f"{chunk.document_id}-")
        assert {
            key: value for key, value in chunk.metadata.items() if key != "header_path"
        } == expected[chunk.document_id]


def test_rejects_reader_path_outside_pinned_release(tmp_path: Path) -> None:
    """A generic reader result may not redirect a manifest document elsewhere."""

    release = tmp_path / "release"
    _, expected = _write_corpus(release)
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    reader = MagicMock()
    filename = min(expected)
    reader.load.return_value = LoadedDocuments(
        documents=(
            Document(
                text="wrong",
                metadata={FILE_NAME: filename, "file_path": str(outside)},
            ),
        )
    )

    with pytest.raises(CorpusValidationError, match="another path"):
        PortfolioReader(text_reader=reader).load(release)
