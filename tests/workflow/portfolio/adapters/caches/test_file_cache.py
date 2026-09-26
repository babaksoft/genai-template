"""Tests for validated atomic filesystem artifact caching."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from genai_template.workflow.portfolio.adapters.caches import (
    CACHE_SCHEMA_VERSION,
    FilesystemArtifactCache,
)
from genai_template.workflow.portfolio.artifacts import (
    canonical_json_bytes,
    sha256_canonical_json,
)
from genai_template.workflow.portfolio.domain import (
    ArtifactCacheError,
    ArtifactProvenance,
    CachedArtifact,
    ComponentSummary,
    EvidenceSection,
    TokenUsage,
)


def _summary() -> ComponentSummary:
    """Build a deterministic valid component summary.

    Returns:
        Valid component output.
    """

    section = EvidenceSection(
        content=("Observed behavior.",), evidence_paths=("src/a.py",)
    )
    return ComponentSummary(
        responsibilities=section,
        important_abstractions=section,
        behavior=section,
        constraints=section,
        testing_evidence=section,
    )


def _artifact(*, fingerprint: str = "1" * 64) -> CachedArtifact:
    """Build a valid cache envelope.

    Args:
        fingerprint:
            Generation fingerprint for the envelope.

    Returns:
        Valid component cache artifact.
    """

    output = _summary().model_dump(mode="json")
    return CachedArtifact(
        cache_schema_version=CACHE_SCHEMA_VERSION,
        provenance=ArtifactProvenance(
            artifact_kind="component",
            generation_fingerprint=fingerprint,
            source_fingerprint="2" * 64,
            unit_input_fingerprint="3" * 64,
            prompt_id="portfolio-component-v1",
            prompt_hash="4" * 64,
            output_schema_version="v1",
            provider="ollama",
            model="test-model",
            temperature=0,
        ),
        structured_output=output,
        output_hash=sha256_canonical_json(output),
        token_usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def _get(cache: FilesystemArtifactCache, fingerprint: str) -> CachedArtifact | None:
    """Load a component entry with the standard test expectations.

    Args:
        cache:
            Filesystem cache under test.
        fingerprint:
            Cache key to load.

    Returns:
        Validated artifact or null.
    """

    return cache.get(
        fingerprint,
        expected_kind="component",
        output_schema_version="v1",
        output_type=ComponentSummary,
    )


def test_cache_miss_write_hit_and_repeated_hit(tmp_path: Path) -> None:
    """Atomic writes become repeatable validated canonical cache hits."""

    cache = FilesystemArtifactCache(tmp_path / "cache")
    artifact = _artifact()

    assert _get(cache, artifact.provenance.generation_fingerprint) is None
    cache.put(artifact, output_type=ComponentSummary)

    assert _get(cache, artifact.provenance.generation_fingerprint) == artifact
    assert _get(cache, artifact.provenance.generation_fingerprint) == artifact
    entry = next((tmp_path / "cache").iterdir())
    assert entry.read_bytes() == canonical_json_bytes(artifact)


@pytest.mark.parametrize(
    "mutation, reason",
    [
        (
            lambda value: value.update(cache_schema_version="v2"),
            "cache-schema-mismatch",
        ),
        (
            lambda value: value["provenance"].update(generation_fingerprint="9" * 64),
            "key-mismatch",
        ),
        (
            lambda value: (
                value["provenance"].update(artifact_kind="overview"),
                value["provenance"].update(unit_input_fingerprint=None),
            ),
            "kind-mismatch",
        ),
        (
            lambda value: value["provenance"].update(output_schema_version="v2"),
            "output-schema-mismatch",
        ),
        (lambda value: value.update(output_hash="8" * 64), "output-hash-mismatch"),
    ],
)
def test_cache_rejects_mismatched_envelopes(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], object],
    reason: str,
) -> None:
    """Existing mismatched cache entries fail instead of becoming misses.

    Args:
        tmp_path:
            Temporary test directory.
        mutation:
            Envelope mutation applied before writing.
        reason:
            Expected stable cache failure reason.
    """

    artifact = _artifact()
    value = artifact.model_dump(mode="json")
    mutation(value)
    root = tmp_path / "cache"
    root.mkdir()
    (root / f"{'1' * 64}.json").write_bytes(canonical_json_bytes(value))

    with pytest.raises(ArtifactCacheError) as caught:
        _get(FilesystemArtifactCache(root), "1" * 64)

    assert caught.value.reason == reason


def test_cache_rejects_truncated_and_noncanonical_json(tmp_path: Path) -> None:
    """Truncated and noncanonical existing entries are explicit failures."""

    root = tmp_path / "cache"
    root.mkdir()
    path = root / f"{'1' * 64}.json"
    cache = FilesystemArtifactCache(root)
    path.write_text('{"broken":', encoding="utf-8")
    with pytest.raises(ArtifactCacheError, match="valid artifact"):
        _get(cache, "1" * 64)

    path.write_text(json.dumps(_artifact().model_dump(mode="json")), encoding="utf-8")
    with pytest.raises(ArtifactCacheError) as caught:
        _get(cache, "1" * 64)
    assert caught.value.reason == "noncanonical-envelope"


def test_failed_atomic_replace_leaves_no_entry_or_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed atomic switch leaves no partial reusable artifact."""

    cache = FilesystemArtifactCache(tmp_path / "cache")

    def fail_replace(source: Path, destination: Path) -> None:
        """Simulate a filesystem failure at the atomic switch.

        Args:
            source:
                Temporary source path.
            destination:
                Final destination path.

        Raises:
            OSError:
                Always, to simulate replacement failure.
        """

        raise OSError("simulated")

    monkeypatch.setattr(
        "genai_template.workflow.portfolio.adapters.caches.file_cache.os.replace",
        fail_replace,
    )
    with pytest.raises(ArtifactCacheError) as caught:
        cache.put(_artifact(), output_type=ComponentSummary)

    assert caught.value.reason == "write-failure"
    assert list((tmp_path / "cache").iterdir()) == []
