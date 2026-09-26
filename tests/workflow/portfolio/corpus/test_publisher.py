"""Tests for immutable release creation and atomic corpus pointer switching."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
from corpus_fixtures import make_corpus_fixture  # noqa: F401
from corpus_fixtures import (
    CorpusFixture,
)

from genai_template.workflow.portfolio.corpus.manifest import build_corpus_manifest
from genai_template.workflow.portfolio.corpus.publisher import publish_corpus
from genai_template.workflow.portfolio.corpus.validation import CorpusValidationError


def test_first_publication_uses_relative_symlink_and_valid_release(
    tmp_path: Path,
    corpus_fixture: CorpusFixture,
) -> None:
    """First publication exposes the complete release through a relative pointer."""

    publication = tmp_path / "data" / "portfolio"
    result = publish_corpus(
        publication_path=publication,
        manifest=corpus_fixture.manifest,
        documents=corpus_fixture.documents,
        snapshot=corpus_fixture.snapshot,
    )

    assert publication.is_symlink()
    assert not Path(os.readlink(publication)).is_absolute()
    assert publication.resolve() == result.release_path
    assert (publication / "manifest.json").is_file()
    assert not result.release_reused


def test_identical_release_is_reused_without_deleting_old_releases(
    tmp_path: Path,
    corpus_fixture: CorpusFixture,
) -> None:
    """An identical fingerprint release is reused and unrelated releases remain."""

    publication = tmp_path / "data" / "portfolio"
    first = publish_corpus(
        publication_path=publication,
        manifest=corpus_fixture.manifest,
        documents=corpus_fixture.documents,
        snapshot=corpus_fixture.snapshot,
    )
    retained = first.release_path.parent / ("f" * 64)
    retained.mkdir()

    second = publish_corpus(
        publication_path=publication,
        manifest=corpus_fixture.manifest,
        documents=corpus_fixture.documents,
        snapshot=corpus_fixture.snapshot,
    )

    assert second.release_reused
    assert retained.is_dir()


def test_replacement_retains_previous_complete_release(
    tmp_path: Path,
    corpus_fixture: CorpusFixture,
) -> None:
    """A new fingerprint switches the pointer without deleting the old release."""

    publication = tmp_path / "data" / "portfolio"
    first = publish_corpus(
        publication_path=publication,
        manifest=corpus_fixture.manifest,
        documents=corpus_fixture.documents,
        snapshot=corpus_fixture.snapshot,
    )
    original = corpus_fixture.documents[0]
    changed_content = original.content + "Changed.\n"
    changed_document = original.model_copy(
        update={
            "content": changed_content,
            "content_hash": hashlib.sha256(changed_content.encode("utf-8")).hexdigest(),
        }
    )
    changed_documents = (changed_document, *corpus_fixture.documents[1:])
    changed_manifest = build_corpus_manifest(
        project_display_name="Sample",
        snapshot=corpus_fixture.snapshot,
        generation=corpus_fixture.generation,
        documents=changed_documents,
    )

    second = publish_corpus(
        publication_path=publication,
        manifest=changed_manifest,
        documents=changed_documents,
        snapshot=corpus_fixture.snapshot,
    )

    assert first.release_path.is_dir()
    assert second.release_path != first.release_path
    assert publication.resolve() == second.release_path


def test_conflicting_existing_release_is_rejected(
    tmp_path: Path,
    corpus_fixture: CorpusFixture,
) -> None:
    """Fingerprint directories with conflicting bytes are never overwritten."""

    publication = tmp_path / "data" / "portfolio"
    conflict = (
        publication.parent
        / ".portfolio-releases"
        / corpus_fixture.manifest.corpus_fingerprint
    )
    conflict.mkdir(parents=True)
    (conflict / "manifest.json").write_text("conflict", encoding="utf-8")

    with pytest.raises(CorpusValidationError):
        publish_corpus(
            publication_path=publication,
            manifest=corpus_fixture.manifest,
            documents=corpus_fixture.documents,
            snapshot=corpus_fixture.snapshot,
        )

    assert (conflict / "manifest.json").read_text(encoding="utf-8") == "conflict"
    assert not publication.exists()


@pytest.mark.parametrize("target_kind", ["file", "directory", "unmanaged_link"])
def test_existing_unmanaged_publication_target_is_refused(
    tmp_path: Path,
    corpus_fixture: CorpusFixture,
    target_kind: str,
) -> None:
    """Real targets and arbitrary symlinks cannot be replaced."""

    publication = tmp_path / "data" / "portfolio"
    publication.parent.mkdir()
    if target_kind == "file":
        publication.write_text("owned", encoding="utf-8")
    elif target_kind == "directory":
        publication.mkdir()
    else:
        publication.symlink_to("somewhere-else")

    with pytest.raises(CorpusValidationError):
        publish_corpus(
            publication_path=publication,
            manifest=corpus_fixture.manifest,
            documents=corpus_fixture.documents,
            snapshot=corpus_fixture.snapshot,
        )


def test_failed_pointer_switch_preserves_previous_publication(
    tmp_path: Path,
    corpus_fixture: CorpusFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A switch failure leaves the prior complete public release readable."""

    publication = tmp_path / "data" / "portfolio"
    first = publish_corpus(
        publication_path=publication,
        manifest=corpus_fixture.manifest,
        documents=corpus_fixture.documents,
        snapshot=corpus_fixture.snapshot,
    )
    prior_target = os.readlink(publication)

    def fail_replace(source: Path, destination: Path) -> None:
        """Inject an atomic pointer replacement failure.

        Args:
            source:
                Temporary symlink path.
            destination:
                Public pointer path.

        Raises:
            OSError:
                Always, to simulate a filesystem failure.
        """

        raise OSError("injected switch failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        publish_corpus(
            publication_path=publication,
            manifest=corpus_fixture.manifest,
            documents=corpus_fixture.documents,
            snapshot=corpus_fixture.snapshot,
        )

    assert os.readlink(publication) == prior_target
    assert publication.resolve() == first.release_path
    assert (publication / "manifest.json").is_file()
