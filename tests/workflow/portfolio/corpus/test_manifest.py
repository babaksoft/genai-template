"""Tests for stable corpus manifest construction and serialization."""

from __future__ import annotations

import json

from corpus_fixtures import make_corpus_fixture  # noqa: F401
from corpus_fixtures import (
    CorpusFixture,
)

from genai_template.workflow.portfolio.corpus.manifest import (
    build_corpus_manifest,
    calculate_corpus_fingerprint,
    generation_configuration_fingerprint,
    manifest_bytes,
)


def test_manifest_bytes_are_canonical_and_stable(corpus_fixture: CorpusFixture) -> None:
    """Manifest JSON is compact, sorted, UTF-8, and newline terminated."""

    raw = manifest_bytes(corpus_fixture.manifest)

    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert b" " not in raw
    assert raw == (
        json.dumps(
            json.loads(raw),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    assert calculate_corpus_fingerprint(corpus_fixture.manifest) == (
        corpus_fixture.manifest.corpus_fingerprint
    )


def test_manifest_omits_volatile_and_machine_local_values(
    corpus_fixture: CorpusFixture,
) -> None:
    """Stable manifest bytes contain no paths, prices, timeouts, or metrics."""

    raw = manifest_bytes(corpus_fixture.manifest).decode("utf-8")

    assert "timeout_seconds" not in raw
    assert "pricing" not in raw
    assert "cache_hit" not in raw
    assert "latency" not in raw
    assert str(corpus_fixture.generation.locations.cache) not in raw


def test_corpus_fingerprint_changes_with_content_or_stable_provenance(
    corpus_fixture: CorpusFixture,
) -> None:
    """Markdown and stable manifest provenance both affect corpus identity."""

    changed_document = corpus_fixture.documents[0].model_copy(
        update={"content_hash": "f" * 64}
    )
    changed_documents = (changed_document, *corpus_fixture.documents[1:])
    changed_content = build_corpus_manifest(
        project_display_name="Sample",
        snapshot=corpus_fixture.snapshot,
        generation=corpus_fixture.generation,
        documents=changed_documents,
    )
    changed_name = build_corpus_manifest(
        project_display_name="Renamed Sample",
        snapshot=corpus_fixture.snapshot,
        generation=corpus_fixture.generation,
        documents=corpus_fixture.documents,
    )

    assert changed_content.corpus_fingerprint != (
        corpus_fixture.manifest.corpus_fingerprint
    )
    assert changed_name.corpus_fingerprint != corpus_fixture.manifest.corpus_fingerprint


def test_configuration_fingerprint_excludes_transport_and_reporting_settings(
    corpus_fixture: CorpusFixture,
) -> None:
    """Timeout, locations, and prices do not affect content configuration identity."""

    structured = corpus_fixture.generation.structured_generation.model_copy(
        update={"timeout_seconds": 999}
    )
    changed = corpus_fixture.generation.model_copy(
        update={"structured_generation": structured}
    )

    assert generation_configuration_fingerprint(changed) == (
        generation_configuration_fingerprint(corpus_fixture.generation)
    )
