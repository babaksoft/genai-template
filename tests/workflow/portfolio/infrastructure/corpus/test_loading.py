"""Tests for pinned, version-neutral Portfolio corpus consumption."""

from __future__ import annotations

import hashlib
import os
from decimal import Decimal
from pathlib import Path

import pytest
from corpus_fixtures import CorpusFixture, make_corpus_fixture  # noqa: F401

from genai_template.config.settings import REPO_ROOT
from genai_template.workflow.portfolio.domain import (
    CorpusManifest,
    CorpusValidationError,
    ManifestDocument,
)
from genai_template.workflow.portfolio.domain.config import (
    GenerationConfig,
    GenerationInputLimits,
    GenerationLocations,
    InferenceConfig,
    StructuredGenerationConfig,
    TokenPricingConfig,
)
from genai_template.workflow.portfolio.infrastructure.corpus import (
    build_future_corpus_manifest,
    load_corpus,
    loading,
    manifest_bytes,
)


def _config() -> GenerationConfig:
    """Return a complete balanced configuration.

    Returns:
        Independent mutable configuration data.
    """

    return GenerationConfig(
        profile="balanced",
        structured_generation=StructuredGenerationConfig(
            provider="ollama",
            model="test-model",
            inference=InferenceConfig(temperature=0, seed=7),
        ),
        prompt_version="v1",
        output_schema_version="v1",
        input_limits=GenerationInputLimits(max_files=10, max_bytes=1024),
        locations=GenerationLocations(
            cache=REPO_ROOT / "storage/test-project-cache",
            publication=REPO_ROOT / "data/test-project-corpus",
        ),
        pricing=TokenPricingConfig(
            input_per_million_tokens=Decimal(2),
            output_per_million_tokens=Decimal(8),
        ),
    )


def _write_release(
    path: Path, fixture: CorpusFixture, manifest: CorpusManifest | None = None
) -> None:
    """Write a canonical release using fixture document bytes.

    Args:
        path:
            Release directory to create.
        fixture:
            Fixture supplying Markdown bytes.
        manifest:
            Optional v1 manifest; the fixture's v2 manifest is used otherwise.
    """

    path.mkdir(parents=True)
    for document in fixture.documents:
        (path / document.filename).write_text(document.content, encoding="utf-8")
    (path / "manifest.json").write_bytes(manifest_bytes(manifest or fixture.manifest))


def test_loads_two_project_v2_with_distinct_provenance(
    tmp_path: Path,
    corpus_fixture: CorpusFixture,
) -> None:
    """One flat v2 release represents heterogeneous project generation."""

    first = corpus_fixture.manifest.projects[0]
    second = first.model_copy(
        update={
            "project_slug": "second",
            "project_display_name": "Second",
            "repository_url": "https://example.test/second.git",
            "resolved_commit_sha": "c" * 40,
            "source_fingerprint": "d" * 64,
        }
    )
    first_record = corpus_fixture.manifest.documents[0]
    second_content = b"# Second overview\n"
    second_record = ManifestDocument(
        project_slug="second",
        file_name="second--component--overview.md",
        document_type="overview",
        component_id="overview",
        evidence_paths=("README.md",),
        generation_fingerprint="f" * 64,
        artifact_hash="1" * 64,
        content_hash=hashlib.sha256(second_content).hexdigest(),
        byte_size=len(second_content),
    )
    manifest = build_future_corpus_manifest(
        generation=_config(),
        projects=(second, first),
        documents=(second_record, first_record),
    )
    release = tmp_path / "release"
    release.mkdir()
    original_document = next(
        document
        for document in corpus_fixture.documents
        if document.filename == first_record.file_name
    )
    original = original_document.content.encode("utf-8")
    (release / first_record.file_name).write_bytes(original)
    (release / second_record.file_name).write_bytes(second_content)
    (release / "manifest.json").write_bytes(manifest_bytes(manifest))

    loaded = load_corpus(release)

    assert [project.project_slug for project in loaded.manifest.projects] == [
        "sample",
        "second",
    ]
    assert [record.project_slug for record in loaded.documents] == [
        "sample",
        "second",
    ]


def test_publication_switch_during_load_keeps_initial_release(
    tmp_path: Path,
    corpus_fixture: CorpusFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manifest and Markdown reads stay on the initially resolved release."""

    first = tmp_path / "releases" / "first"
    second = tmp_path / "releases" / "second"
    _write_release(first, corpus_fixture)
    _write_release(second, corpus_fixture)
    publication = tmp_path / "portfolio"
    publication.symlink_to(first.relative_to(tmp_path))
    real_read_manifest = loading.read_manifest

    def switch_then_read(path: Path) -> CorpusManifest:
        """Switch the pointer after pinning, then read the pinned manifest.

        Args:
            path:
                Pinned manifest path supplied by the loader.

        Returns:
            Strictly parsed manifest from the pinned release.
        """

        replacement = tmp_path / ".portfolio.next"
        replacement.symlink_to(second.relative_to(tmp_path))
        os.replace(replacement, publication)
        return real_read_manifest(path)

    monkeypatch.setattr(loading, "read_manifest", switch_then_read)

    loaded = load_corpus(publication)

    assert loaded.release_path == first
    assert publication.resolve() == second


@pytest.mark.parametrize("mutation", ["unknown_version", "nested", "symlinked"])
def test_malformed_release_fails_closed(
    tmp_path: Path,
    corpus_fixture: CorpusFixture,
    mutation: str,
) -> None:
    """Unknown manifests and non-regular flat entries are rejected."""

    release = tmp_path / "release"
    _write_release(release, corpus_fixture)
    if mutation == "unknown_version":
        (release / "manifest.json").write_bytes(b'{"manifest_schema_version":99}\n')
    elif mutation == "nested":
        (release / "nested").mkdir()
    else:
        (release / "linked.md").symlink_to(corpus_fixture.documents[0].filename)

    with pytest.raises(CorpusValidationError):
        load_corpus(release)
