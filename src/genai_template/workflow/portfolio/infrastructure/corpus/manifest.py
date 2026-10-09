"""Canonical construction and serialization of Portfolio corpus manifests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from genai_template.workflow.portfolio.domain.config import GenerationConfig
from genai_template.workflow.portfolio.domain.corpus import (
    CorpusManifestV2,
    ManifestDocumentV2,
    ManifestProject,
    ManifestPrompt,
    RenderedDocument,
    VersionedCorpusManifest,
)
from genai_template.workflow.portfolio.domain.snapshot import RepositorySnapshot
from genai_template.workflow.portfolio.infrastructure.corpus.renderers import (
    build_document_filename,
)
from genai_template.workflow.portfolio.infrastructure.fingerprints import (
    calculate_corpus_fingerprint,
    canonical_json_bytes,
    generation_configuration_fingerprint,
)
from genai_template.workflow.portfolio.infrastructure.generation.prompts import (
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


def build_corpus_manifest(
    *,
    project_display_name: str,
    snapshot: RepositorySnapshot,
    generation: GenerationConfig,
    documents: Sequence[RenderedDocument],
) -> CorpusManifestV2:
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
        Complete immutable v2 corpus manifest.

    Raises:
        ValueError:
            If documents are empty, duplicated, or have invalid ownership.
    """

    records = tuple(
        _document_record(document, project_slug=snapshot.project_slug)
        for document in documents
    )
    filenames = tuple(record.filename for record in records)
    if not records:
        raise ValueError("a corpus manifest requires at least one document")
    if len(filenames) != len(set(filenames)):
        raise ValueError("manifest document filenames must be unique")
    if filenames != tuple(sorted(filenames)):
        records = tuple(sorted(records, key=lambda record: record.filename))

    project = ManifestProject(
        project_slug=snapshot.project_slug,
        project_display_name=project_display_name,
        repository_url=snapshot.repository_url,
        requested_ref=snapshot.requested_ref,
        resolved_commit_sha=snapshot.resolved_commit_sha,
        source_fingerprint=snapshot.source_fingerprint,
        generation_profile=generation.profile,
        prompt_version=generation.prompt_version,
        prompts=tuple(
            ManifestPrompt(prompt_id=prompt.prompt_id, prompt_hash=prompt.prompt_hash)
            for prompt in _PROMPTS
        ),
        output_schema_version=generation.output_schema_version,
        provider=generation.structured_generation.provider,
        model=generation.structured_generation.model,
        configuration_fingerprint=generation_configuration_fingerprint(generation),
    )
    return build_corpus_manifest_v2(projects=(project,), documents=records)


def build_corpus_manifest_v2(
    *,
    projects: Sequence[ManifestProject],
    documents: Sequence[ManifestDocumentV2],
) -> CorpusManifestV2:
    """Build a canonical v2 manifest from one or more project projections.

    Args:
        projects:
            Project provenance entries to order by project slug.
        documents:
            Project-owned document records to order by filename.

    Returns:
        Complete immutable v2 corpus manifest.

    Raises:
        ValueError:
            If projects or documents are empty, duplicated, or reference an
            unknown project.
    """

    ordered_projects = tuple(sorted(projects, key=lambda project: project.project_slug))
    ordered_documents = tuple(sorted(documents, key=lambda document: document.filename))
    project_slugs = tuple(project.project_slug for project in ordered_projects)
    filenames = tuple(document.filename for document in ordered_documents)

    if not ordered_projects:
        raise ValueError("a corpus manifest requires at least one project")
    if len(project_slugs) != len(set(project_slugs)):
        raise ValueError("manifest project slugs must be unique")
    if not ordered_documents:
        raise ValueError("a corpus manifest requires at least one document")
    if len(filenames) != len(set(filenames)):
        raise ValueError("manifest document filenames must be unique")
    if any(
        document.project_slug not in set(project_slugs)
        for document in ordered_documents
    ):
        raise ValueError("manifest documents must reference a known project")

    for document in ordered_documents:
        try:
            expected_filename = build_document_filename(
                document.project_slug,
                document.document_type,
                component_id=document.component_id,
            )
        except ValueError as exc:
            raise ValueError("manifest document identity is invalid") from exc
        if document.filename != expected_filename:
            raise ValueError("manifest document filename does not match its owner")

    data: dict[str, Any] = {
        "manifest_schema_version": 2,
        "projects": tuple(
            project.model_dump(mode="json") for project in ordered_projects
        ),
        "documents": tuple(
            document.model_dump(mode="json") for document in ordered_documents
        ),
    }
    data["corpus_fingerprint"] = calculate_corpus_fingerprint(data, ordered_documents)
    return CorpusManifestV2.model_validate(data)


def manifest_bytes(manifest: VersionedCorpusManifest) -> bytes:
    """Serialize a manifest as compact canonical JSON with one final newline.

    Args:
        manifest:
            Validated manifest to serialize.

    Returns:
        Canonical UTF-8 JSON bytes ending in exactly one newline.
    """

    return canonical_json_bytes(manifest) + b"\n"


def _document_record(
    document: RenderedDocument, *, project_slug: str
) -> ManifestDocumentV2:
    """Project a rendered document into stable manifest metadata.

    Args:
        document:
            Validated rendered document.
        project_slug:
            Slug of the project that owns the document.

    Returns:
        Stable document manifest record.
    """

    return ManifestDocumentV2(
        project_slug=project_slug,
        filename=document.filename,
        document_type=document.document_type,
        component_id=document.component_id,
        evidence_paths=document.evidence_paths,
        generation_fingerprint=document.generation_fingerprint,
        artifact_hash=document.artifact_hash,
        content_hash=document.content_hash,
        byte_size=len(document.content.encode("utf-8")),
    )
