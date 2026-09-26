"""Canonical construction and serialization of Portfolio corpus manifests."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from genai_template.workflow.portfolio.artifacts.fingerprints import (
    canonical_json_bytes,
    sha256_canonical_json,
)
from genai_template.workflow.portfolio.config.models import GenerationConfig
from genai_template.workflow.portfolio.domain.generation import RenderedDocument
from genai_template.workflow.portfolio.domain.manifest import (
    CorpusManifest,
    ManifestDocument,
    ManifestPrompt,
)
from genai_template.workflow.portfolio.domain.snapshot import RepositorySnapshot
from genai_template.workflow.portfolio.generation.prompts import (
    ARCHITECTURE_PROMPT,
    COMPONENT_PROMPT,
    OVERVIEW_PROMPT,
    TESTING_OPERATIONS_PROMPT,
)

_PROMPTS = (
    COMPONENT_PROMPT,
    OVERVIEW_PROMPT,
    ARCHITECTURE_PROMPT,
    TESTING_OPERATIONS_PROMPT,
)


def generation_configuration_fingerprint(config: GenerationConfig) -> str:
    """Calculate a stable identity for content-affecting generation settings.

    Transport timeouts, storage locations, and pricing are excluded because they
    cannot change generated content. Prompt templates and source identities have
    their own explicit manifest fields.

    Args:
        config:
            Validated corpus-generation configuration.

    Returns:
        Lowercase hexadecimal SHA-256 configuration identity.
    """

    projection = {
        "configuration_schema": "portfolio-generation-configuration-v1",
        "profile": config.profile,
        "structured_generation": {
            "provider": config.structured_generation.provider,
            "model": config.structured_generation.model,
            "inference": config.structured_generation.inference.model_dump(mode="json"),
        },
        "prompt_version": config.prompt_version,
        "output_schema_version": config.output_schema_version,
        "project_context": config.project_context.model_dump(mode="json"),
        "input_limits": config.input_limits.model_dump(mode="json"),
    }
    return sha256_canonical_json(projection)


def build_corpus_manifest(
    *,
    project_display_name: str,
    snapshot: RepositorySnapshot,
    generation: GenerationConfig,
    documents: Sequence[RenderedDocument],
) -> CorpusManifest:
    """Build a canonical manifest and its non-circular corpus fingerprint.

    Args:
        project_display_name:
            Human-readable configured project name.
        snapshot:
            Exact immutable source snapshot used for generation.
        generation:
            Validated generation settings.
        documents:
            Complete rendered balanced-profile documents.

    Returns:
        Complete immutable corpus manifest.

    Raises:
        ValueError:
            If documents are empty, duplicated, or not in canonical order.
    """

    records = tuple(_document_record(document) for document in documents)
    filenames = tuple(record.filename for record in records)
    if not records:
        raise ValueError("a corpus manifest requires at least one document")
    if len(filenames) != len(set(filenames)):
        raise ValueError("manifest document filenames must be unique")
    if filenames != tuple(sorted(filenames)):
        records = tuple(sorted(records, key=lambda record: record.filename))

    data: dict[str, Any] = {
        "manifest_schema_version": 1,
        "project_slug": snapshot.project_slug,
        "project_display_name": project_display_name,
        "repository_url": snapshot.repository_url,
        "requested_ref": snapshot.requested_ref,
        "resolved_commit_sha": snapshot.resolved_commit_sha,
        "source_fingerprint": snapshot.source_fingerprint,
        "generation_profile": generation.profile,
        "prompt_version": generation.prompt_version,
        "prompts": tuple(
            ManifestPrompt(
                prompt_id=prompt.prompt_id,
                prompt_hash=prompt.prompt_hash,
            ).model_dump(mode="json")
            for prompt in _PROMPTS
        ),
        "output_schema_version": generation.output_schema_version,
        "provider": generation.structured_generation.provider,
        "model": generation.structured_generation.model,
        "configuration_fingerprint": generation_configuration_fingerprint(generation),
        "documents": tuple(record.model_dump(mode="json") for record in records),
    }
    data["corpus_fingerprint"] = calculate_corpus_fingerprint(data, records)
    return CorpusManifest.model_validate(data)


def calculate_corpus_fingerprint(
    manifest: CorpusManifest | dict[str, Any],
    documents: Sequence[ManifestDocument] | None = None,
) -> str:
    """Calculate the non-circular stable identity of a complete corpus.

    The hash input contains the canonical manifest with ``corpus_fingerprint``
    omitted and an explicit ordered filename/content-hash list. The latter makes
    the byte identity boundary obvious even though hashes also occur in records.

    Args:
        manifest:
            Complete manifest or pre-validation manifest data.
        documents:
            Document records when ``manifest`` does not yet include them.

    Returns:
        Lowercase hexadecimal SHA-256 corpus identity.
    """

    projection = (
        manifest.model_dump(mode="json")
        if isinstance(manifest, CorpusManifest)
        else dict(manifest)
    )
    projection.pop("corpus_fingerprint", None)
    records_value = documents if documents is not None else projection["documents"]
    records = [
        (
            record.model_dump(mode="json")
            if isinstance(record, ManifestDocument)
            else record
        )
        for record in records_value
    ]
    projection["documents"] = records
    ordered_hashes = [
        {"filename": record["filename"], "content_hash": record["content_hash"]}
        for record in sorted(records, key=lambda value: value["filename"])
    ]
    envelope = {
        "fingerprint_schema": "portfolio-corpus-v1",
        "manifest": projection,
        "markdown": ordered_hashes,
    }
    return hashlib.sha256(canonical_json_bytes(envelope)).hexdigest()


def manifest_bytes(manifest: CorpusManifest) -> bytes:
    """Serialize a manifest as compact canonical JSON with one final newline.

    Args:
        manifest:
            Validated manifest to serialize.

    Returns:
        Canonical UTF-8 JSON bytes ending in exactly one newline.
    """

    return canonical_json_bytes(manifest) + b"\n"


def _document_record(document: RenderedDocument) -> ManifestDocument:
    """Project a rendered document into stable manifest metadata.

    Args:
        document:
            Validated rendered document.

    Returns:
        Stable document manifest record.
    """

    return ManifestDocument(
        filename=document.filename,
        document_type=document.document_type,
        component_id=document.component_id,
        evidence_paths=document.evidence_paths,
        generation_fingerprint=document.generation_fingerprint,
        artifact_hash=document.artifact_hash,
        content_hash=document.content_hash,
        byte_size=len(document.content.encode("utf-8")),
    )
