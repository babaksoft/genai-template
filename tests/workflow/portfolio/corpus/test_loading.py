"""Tests for pinned, version-neutral Portfolio corpus consumption."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
from corpus_fixtures import CorpusFixture, make_corpus_fixture  # noqa: F401

from genai_template.workflow.portfolio.corpus import loading
from genai_template.workflow.portfolio.corpus.loading import (
    load_corpus,
    normalize_manifest,
)
from genai_template.workflow.portfolio.corpus.manifest import (
    build_corpus_manifest_v2,
    calculate_corpus_fingerprint,
    manifest_bytes,
)
from genai_template.workflow.portfolio.corpus.validation import CorpusValidationError
from genai_template.workflow.portfolio.domain.corpus import (
    CorpusManifest,
    ManifestDocumentV2,
    VersionedCorpusManifest,
)


def _legacy_manifest(fixture: CorpusFixture) -> CorpusManifest:
    """Project a current one-project fixture into the unchanged v1 contract.

    Args:
        fixture:
            Current consistent corpus fixture.

    Returns:
        Canonical legacy manifest with its original v1 fingerprint algorithm.
    """

    project = fixture.manifest.projects[0]
    data = {
        "manifest_schema_version": 1,
        **project.model_dump(mode="json"),
        "documents": tuple(
            {
                key: value
                for key, value in record.model_dump(mode="json").items()
                if key != "project_slug"
            }
            for record in fixture.manifest.documents
        ),
    }
    data["corpus_fingerprint"] = calculate_corpus_fingerprint(data)
    return CorpusManifest.model_validate(data)


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


def test_v1_normalization_preserves_original_fingerprint(
    corpus_fixture: CorpusFixture,
) -> None:
    """Legacy bytes normalize into one project without changing identity."""

    legacy = _legacy_manifest(corpus_fixture)

    normalized = normalize_manifest(legacy)

    assert normalized.source_schema_version == 1
    assert normalized.corpus_fingerprint == legacy.corpus_fingerprint
    assert normalized.projects[0].project_slug == legacy.project_slug
    assert {record.project_slug for record in normalized.documents} == {"sample"}


def test_loads_canonical_legacy_release(
    tmp_path: Path,
    corpus_fixture: CorpusFixture,
) -> None:
    """A byte-identical v1 release remains valid for consumers."""

    legacy = _legacy_manifest(corpus_fixture)
    release = tmp_path / "release"
    _write_release(release, corpus_fixture, legacy)
    original_bytes = (release / "manifest.json").read_bytes()

    loaded = load_corpus(release)

    assert loaded.manifest.source_schema_version == 1
    assert loaded.manifest.corpus_fingerprint == legacy.corpus_fingerprint
    assert (release / "manifest.json").read_bytes() == original_bytes


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
            "model": "other-model",
            "configuration_fingerprint": "e" * 64,
        }
    )
    first_record = corpus_fixture.manifest.documents[0]
    second_content = b"# Second overview\n"
    second_record = ManifestDocumentV2(
        project_slug="second",
        filename="second--overview.md",
        document_type="overview",
        component_id=None,
        evidence_paths=("README.md",),
        generation_fingerprint="f" * 64,
        artifact_hash="1" * 64,
        content_hash=hashlib.sha256(second_content).hexdigest(),
        byte_size=len(second_content),
    )
    manifest = build_corpus_manifest_v2(
        projects=(second, first),
        documents=(second_record, first_record),
    )
    release = tmp_path / "release"
    release.mkdir()
    original_document = next(
        document
        for document in corpus_fixture.documents
        if document.filename == first_record.filename
    )
    original = original_document.content.encode("utf-8")
    (release / first_record.filename).write_bytes(original)
    (release / second_record.filename).write_bytes(second_content)
    (release / "manifest.json").write_bytes(manifest_bytes(manifest))

    loaded = load_corpus(release)

    assert [project.project_slug for project in loaded.manifest.projects] == [
        "sample",
        "second",
    ]
    assert loaded.manifest.projects[1].model == "other-model"
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

    def switch_then_read(path: Path) -> VersionedCorpusManifest:
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
