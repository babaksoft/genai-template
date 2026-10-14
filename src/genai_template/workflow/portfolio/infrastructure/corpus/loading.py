"""Pinned, repository-independent loading of published Portfolio corpora."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath

from genai_template.workflow.portfolio.domain import (
    CorpusManifest,
    CorpusValidationError,
    LoadedCorpus,
)
from genai_template.workflow.portfolio.infrastructure.corpus.renderers import (
    build_document_file_name,
)
from genai_template.workflow.portfolio.infrastructure.corpus.validation import (
    read_manifest,
)
from genai_template.workflow.portfolio.infrastructure.fingerprints import (
    calculate_corpus_fingerprint,
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
    _validate_consumer_contract(manifest)

    expected_names = {
        "manifest.json",
        *(record.file_name for record in manifest.documents),
    }
    if set(entries) != expected_names:
        raise CorpusValidationError("corpus file set does not match its manifest")

    for record in manifest.documents:
        try:
            raw = entries[record.file_name].read_bytes()
        except OSError as exc:
            raise CorpusValidationError(
                f"document cannot be read: {record.file_name}"
            ) from exc
        if len(raw) != record.byte_size:
            raise CorpusValidationError(
                f"document byte size does not match manifest: {record.file_name}"
            )
        if hashlib.sha256(raw).hexdigest() != record.content_hash:
            raise CorpusValidationError(
                f"document content hash does not match manifest: {record.file_name}"
            )

    if calculate_corpus_fingerprint(manifest) != manifest.corpus_fingerprint:
        raise CorpusValidationError("corpus fingerprint does not match manifest")

    return LoadedCorpus(
        release_path=pinned_path,
        manifest=manifest,
        documents=manifest.documents,
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


def _validate_consumer_contract(manifest: CorpusManifest) -> None:
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

    file_names = tuple(record.file_name for record in manifest.documents)
    if file_names != tuple(sorted(file_names)) or len(file_names) != len(
        set(file_names)
    ):
        raise CorpusValidationError("manifest file names must be unique and sorted")

    known_projects = set(project_slugs)
    for record in manifest.documents:
        if record.project_slug not in known_projects:
            raise CorpusValidationError("document references an unknown project")
        pure_name = PurePosixPath(record.file_name)
        if pure_name.name != record.file_name or pure_name.suffix != ".md":
            raise CorpusValidationError("manifest contains an unsafe file name")

        try:
            expected = build_document_file_name(
                record.project_slug,
                record.component_id,
            )
        except ValueError as exc:
            raise CorpusValidationError("document identity is invalid") from exc
        if record.file_name != expected:
            raise CorpusValidationError("document file name does not match its owner")

        if not record.evidence_paths:
            raise CorpusValidationError("every document requires evidence")
        if record.evidence_paths != tuple(sorted(set(record.evidence_paths))):
            raise CorpusValidationError("document evidence must be unique and sorted")
