"""Tests for complete staged-corpus validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from corpus_fixtures import make_corpus_fixture  # noqa: F401
from corpus_fixtures import (
    CorpusFixture,
)

from genai_template.workflow.portfolio.corpus.manifest import manifest_bytes
from genai_template.workflow.portfolio.corpus.validation import (
    CorpusValidationError,
    validate_corpus_directory,
)


def _write_corpus(path: Path, fixture: CorpusFixture) -> None:
    """Write one valid corpus directory for mutation by a test.

    Args:
        path:
            Empty target directory.
        fixture:
            Consistent corpus fixture.
    """

    path.mkdir()
    for document in fixture.documents:
        (path / document.filename).write_text(document.content, encoding="utf-8")
    (path / "manifest.json").write_bytes(manifest_bytes(fixture.manifest))


def test_complete_corpus_validates(
    tmp_path: Path,
    corpus_fixture: CorpusFixture,
) -> None:
    """A complete flat corpus agrees with its source and stable fingerprint."""

    corpus = tmp_path / "corpus"
    _write_corpus(corpus, corpus_fixture)

    result = validate_corpus_directory(corpus, corpus_fixture.snapshot)

    assert result == corpus_fixture.manifest


@pytest.mark.parametrize("mutation", ["extra", "missing", "nested", "content"])
def test_file_set_and_bytes_are_strictly_validated(
    tmp_path: Path,
    corpus_fixture: CorpusFixture,
    mutation: str,
) -> None:
    """Extra, missing, nested, and modified files are all rejected."""

    corpus = tmp_path / "corpus"
    _write_corpus(corpus, corpus_fixture)
    filename = corpus_fixture.manifest.documents[0].filename
    if mutation == "extra":
        (corpus / "extra.md").write_text("extra", encoding="utf-8")
    elif mutation == "missing":
        (corpus / filename).unlink()
    elif mutation == "nested":
        (corpus / "nested").mkdir()
    else:
        (corpus / filename).write_text("changed", encoding="utf-8")

    with pytest.raises(CorpusValidationError):
        validate_corpus_directory(corpus, corpus_fixture.snapshot)


def test_noncanonical_manifest_is_rejected(
    tmp_path: Path,
    corpus_fixture: CorpusFixture,
) -> None:
    """Semantically valid but pretty-printed manifest JSON is not accepted."""

    corpus = tmp_path / "corpus"
    _write_corpus(corpus, corpus_fixture)
    value = corpus_fixture.manifest.model_dump(mode="json")
    (corpus / "manifest.json").write_text(json.dumps(value, indent=2), encoding="utf-8")

    with pytest.raises(CorpusValidationError, match="not canonical"):
        validate_corpus_directory(corpus, corpus_fixture.snapshot)


def test_evidence_outside_supplied_snapshot_is_rejected(
    tmp_path: Path,
    corpus_fixture: CorpusFixture,
) -> None:
    """Manifest evidence must belong to the exact supplied snapshot."""

    corpus = tmp_path / "corpus"
    _write_corpus(corpus, corpus_fixture)
    record = corpus_fixture.manifest.documents[0].model_copy(
        update={"evidence_paths": ("missing.py",)}
    )
    manifest = corpus_fixture.manifest.model_copy(
        update={"documents": (record, *corpus_fixture.manifest.documents[1:])}
    )
    (corpus / "manifest.json").write_bytes(manifest_bytes(manifest))

    with pytest.raises(CorpusValidationError, match="outside the snapshot"):
        validate_corpus_directory(corpus, corpus_fixture.snapshot)
