"""Pinned, repository-independent loading of published Portfolio corpora."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath

from genai_template.workflow.portfolio.domain import (
    CorpusManifestV2,
    CorpusValidationError,
    LoadedCorpus,
    ManifestDocumentV2,
    ManifestProject,
    NormalizedCorpusManifest,
    VersionedCorpusManifest,
)
from genai_template.workflow.portfolio.infrastructure.corpus.renderers import (
    build_document_filename,
)
from genai_template.workflow.portfolio.infrastructure.corpus.validation import (
    read_manifest,
)
from genai_template.workflow.portfolio.infrastructure.fingerprints import (
    calculate_corpus_fingerprint,
)


def normalize_manifest(manifest: VersionedCorpusManifest) -> NormalizedCorpusManifest:
    """Normalize a v1 or v2 manifest.

    Args:
        manifest:
            Strictly parsed versioned manifest.

    Returns:
        Immutable version-neutral consumer projection.
    """

    if isinstance(manifest, CorpusManifestV2):
        return NormalizedCorpusManifest(
            source_schema_version=2,
            projects=manifest.projects,
            documents=manifest.documents,
            corpus_fingerprint=manifest.corpus_fingerprint,
        )

    project = ManifestProject(
        project_slug=manifest.project_slug,
        project_display_name=manifest.project_display_name,
        repository_url=manifest.repository_url,
        requested_ref=manifest.requested_ref,
        resolved_commit_sha=manifest.resolved_commit_sha,
        source_fingerprint=manifest.source_fingerprint,
        generation_profile=manifest.generation_profile,
        prompt_version=manifest.prompt_version,
        prompts=manifest.prompts,
        output_schema_version=manifest.output_schema_version,
        provider=manifest.provider,
        model=manifest.model,
        configuration_fingerprint=manifest.configuration_fingerprint,
    )
    documents = tuple(
        ManifestDocumentV2(
            project_slug=manifest.project_slug,
            **record.model_dump(mode="python"),
        )
        for record in manifest.documents
    )

    return NormalizedCorpusManifest(
        source_schema_version=1,
        projects=(project,),
        documents=documents,
        corpus_fingerprint=manifest.corpus_fingerprint,
    )


def load_corpus(source_path: Path) -> LoadedCorpus:
    """Pin, validate, and normalize one published Portfolio corpus.

    The supplied publication pointer is resolved once before any corpus bytes are
    read. All subsequent operations use that pinned directory, so an atomic pointer
    switch cannot mix releases.

    Args:
        source_path:
            Published corpus symlink or a real release directory.

    Returns:
        Pinned release location and normalized stable provenance.

    Raises:
        CorpusValidationError:
            If the source, manifest, directory shape, bytes, or fingerprint is
            invalid.
    """

    try:
        pinned_path = source_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CorpusValidationError("corpus source cannot be resolved") from exc
    if not pinned_path.is_dir() or pinned_path.is_symlink():
        raise CorpusValidationError("corpus source must resolve to a real directory")

    entries = _read_flat_entries(pinned_path)
    manifest_path = entries.get("manifest.json")
    if manifest_path is None:
        raise CorpusValidationError("corpus is missing manifest.json")
    manifest = read_manifest(manifest_path)
    normalized = normalize_manifest(manifest)
    _validate_consumer_contract(normalized)

    expected_names = {
        "manifest.json",
        *(record.filename for record in normalized.documents),
    }
    if set(entries) != expected_names:
        raise CorpusValidationError("corpus file set does not match its manifest")

    for record in normalized.documents:
        try:
            raw = entries[record.filename].read_bytes()
        except OSError as exc:
            raise CorpusValidationError(
                f"document cannot be read: {record.filename}"
            ) from exc
        if len(raw) != record.byte_size:
            raise CorpusValidationError(
                f"document byte size does not match manifest: {record.filename}"
            )
        if hashlib.sha256(raw).hexdigest() != record.content_hash:
            raise CorpusValidationError(
                f"document content hash does not match manifest: {record.filename}"
            )

    if calculate_corpus_fingerprint(manifest) != manifest.corpus_fingerprint:
        raise CorpusValidationError("corpus fingerprint does not match manifest")
    return LoadedCorpus(
        release_path=pinned_path,
        manifest=normalized,
        documents=normalized.documents,
    )


def _read_flat_entries(directory: Path) -> dict[str, Path]:
    """Read and validate the complete immediate release directory.

    Args:
        directory:
            Pinned real release directory.

    Returns:
        Entry names mapped to their pinned paths.

    Raises:
        CorpusValidationError:
            If entries cannot be listed or are nested, special, or symlinked.
    """

    try:
        entries = tuple(directory.iterdir())
    except OSError as exc:
        raise CorpusValidationError("corpus directory cannot be read") from exc

    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise CorpusValidationError("corpus must contain only flat regular files")

    return {entry.name: entry for entry in entries}


def _validate_consumer_contract(manifest: NormalizedCorpusManifest) -> None:
    """Validate normalized ordering, ownership, and safe identities.

    Args:
        manifest:
            Normalized consumer projection.

    Raises:
        CorpusValidationError:
            If project or document identities are ambiguous or unsafe.
    """

    project_slugs = tuple(project.project_slug for project in manifest.projects)
    if project_slugs != tuple(sorted(project_slugs)) or len(project_slugs) != len(
        set(project_slugs)
    ):
        raise CorpusValidationError("manifest projects must be unique and sorted")

    filenames = tuple(record.filename for record in manifest.documents)
    if filenames != tuple(sorted(filenames)) or len(filenames) != len(set(filenames)):
        raise CorpusValidationError("manifest filenames must be unique and sorted")

    known_projects = set(project_slugs)
    for record in manifest.documents:
        if record.project_slug not in known_projects:
            raise CorpusValidationError("document references an unknown project")
        pure_name = PurePosixPath(record.filename)
        if pure_name.name != record.filename or pure_name.suffix != ".md":
            raise CorpusValidationError("manifest contains an unsafe filename")

        try:
            expected = build_document_filename(
                record.project_slug,
                record.document_type,
                component_id=record.component_id,
            )
        except ValueError as exc:
            raise CorpusValidationError("document identity is invalid") from exc
        if record.filename != expected:
            raise CorpusValidationError("document filename does not match its owner")

        if not record.evidence_paths:
            raise CorpusValidationError("every document requires evidence")
        if record.evidence_paths != tuple(sorted(set(record.evidence_paths))):
            raise CorpusValidationError("document evidence must be unique and sorted")
