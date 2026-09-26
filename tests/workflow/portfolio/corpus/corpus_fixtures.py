"""Shared fixtures for whole-corpus manifest and publication tests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest

from genai_template.workflow.portfolio.config import load_portfolio_config
from genai_template.workflow.portfolio.config.models import GenerationConfig
from genai_template.workflow.portfolio.corpus.manifest import build_corpus_manifest
from genai_template.workflow.portfolio.domain import (
    ArtifactKind,
    CorpusManifest,
    RenderedDocument,
    RepositorySnapshot,
    SnapshotFile,
)


@dataclass(frozen=True)
class CorpusFixture:
    """Internally consistent corpus values for focused tests.

    Attributes:
        snapshot:
            Source snapshot supplying all evidence paths.
        generation:
            Valid balanced generation configuration.
        documents:
            Complete rendered document set.
        manifest:
            Manifest derived from the other fixture values.
    """

    snapshot: RepositorySnapshot
    generation: GenerationConfig
    documents: tuple[RenderedDocument, ...]
    manifest: CorpusManifest


@pytest.fixture(name="corpus_fixture")
def make_corpus_fixture() -> CorpusFixture:
    """Build a small complete balanced corpus fixture.

    Returns:
        Consistent snapshot, generation settings, documents, and manifest.
    """

    source = b"source\n"
    snapshot = RepositorySnapshot(
        project_slug="sample",
        repository_url="https://example.test/sample.git",
        requested_ref="main",
        resolved_commit_sha="a" * 40,
        source_fingerprint="b" * 64,
        files=(
            SnapshotFile(
                path="README.md",
                text=source.decode("utf-8"),
                content_hash=hashlib.sha256(source).hexdigest(),
                byte_size=len(source),
            ),
        ),
    )
    config = load_portfolio_config(
        Path("src/genai_template/workflow/portfolio/config/profiles/local.yml")
    ).require_generation()
    specifications: tuple[tuple[str, ArtifactKind, str | None, str], ...] = (
        ("sample--overview.md", "overview", None, "1"),
        ("sample--architecture.md", "architecture", None, "2"),
        ("sample--testing-operations.md", "testing_operations", None, "3"),
        ("sample--component--api.md", "component", "api", "4"),
    )
    documents = tuple(
        _document(filename, document_type, component_id, character)
        for filename, document_type, component_id, character in specifications
    )
    manifest = build_corpus_manifest(
        project_display_name="Sample",
        snapshot=snapshot,
        generation=config,
        documents=documents,
    )
    return CorpusFixture(snapshot, config, documents, manifest)


def _document(
    filename: str,
    document_type: ArtifactKind,
    component_id: str | None,
    character: str,
) -> RenderedDocument:
    """Build one internally consistent rendered document.

    Args:
        filename:
            Expected safe Markdown filename.
        document_type:
            Logical artifact type.
        component_id:
            Component identifier when applicable.
        character:
            Character used for deterministic test hashes.

    Returns:
        Rendered document with a hash of its actual content.
    """

    content = f"# {document_type}\n\nEvidence: `README.md`\n"
    return RenderedDocument(
        filename=filename,
        document_type=document_type,
        component_id=component_id,
        generation_fingerprint=character * 64,
        artifact_hash=character * 64,
        content=content,
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        evidence_paths=("README.md",),
    )
