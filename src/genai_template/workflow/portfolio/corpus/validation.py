"""Whole-corpus validation before and after filesystem publication."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from genai_template.workflow.portfolio.corpus.manifest import (
    calculate_corpus_fingerprint,
    manifest_bytes,
)
from genai_template.workflow.portfolio.corpus.renderers import build_document_filename
from genai_template.workflow.portfolio.domain.generation import RenderedDocument
from genai_template.workflow.portfolio.domain.manifest import CorpusManifest
from genai_template.workflow.portfolio.domain.snapshot import RepositorySnapshot

_SAFE_FILENAME = re.compile(
    r"^[a-z0-9]+(?:-[a-z0-9]+)*(?:--[a-z0-9]+(?:-[a-z0-9]+)*)+\.md$"
)


class CorpusValidationError(RuntimeError):
    """Failure to validate a complete corpus as one publication unit."""


def validate_rendered_corpus(
    manifest: CorpusManifest,
    documents: Sequence[RenderedDocument],
    snapshot: RepositorySnapshot,
) -> None:
    """Validate a complete rendered corpus before filesystem staging.

    Args:
        manifest:
            Stable manifest proposed for publication.
        documents:
            Complete rendered Markdown document set.
        snapshot:
            Exact immutable snapshot defining valid evidence paths.

    Raises:
        CorpusValidationError:
            If identities, records, content bytes, or evidence disagree.
    """

    _validate_source_identity(manifest, snapshot)
    _validate_document_records(manifest, snapshot)
    records = {record.filename: record for record in manifest.documents}
    supplied = {document.filename: document for document in documents}
    if len(supplied) != len(documents) or supplied.keys() != records.keys():
        raise CorpusValidationError("rendered documents do not match the manifest")
    for filename, document in supplied.items():
        record = records[filename]
        content = document.content.encode("utf-8")
        if (
            document.document_type != record.document_type
            or document.component_id != record.component_id
            or document.generation_fingerprint != record.generation_fingerprint
            or document.artifact_hash != record.artifact_hash
            or document.content_hash != record.content_hash
            or document.evidence_paths != record.evidence_paths
            or len(content) != record.byte_size
            or hashlib.sha256(content).hexdigest() != record.content_hash
        ):
            raise CorpusValidationError(
                "rendered document metadata differs from manifest"
            )
    if calculate_corpus_fingerprint(manifest) != manifest.corpus_fingerprint:
        raise CorpusValidationError("corpus fingerprint does not match manifest")


def read_manifest(path: Path) -> CorpusManifest:
    """Read and strictly validate a canonical on-disk manifest.

    Args:
        path:
            Manifest file to read.

    Returns:
        Parsed immutable corpus manifest.

    Raises:
        CorpusValidationError:
            If bytes are unreadable, invalid, or non-canonical.
    """

    try:
        raw = path.read_bytes()
        value = json.loads(raw)
        manifest = CorpusManifest.model_validate(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise CorpusValidationError("manifest.json is invalid") from exc
    if raw != manifest_bytes(manifest):
        raise CorpusValidationError("manifest.json is not canonical")
    return manifest


def validate_corpus_directory(
    directory: Path,
    snapshot: RepositorySnapshot,
    *,
    expected_manifest: CorpusManifest | None = None,
) -> CorpusManifest:
    """Validate every file and stable identity in a staged or released corpus.

    Args:
        directory:
            Flat corpus directory containing Markdown and ``manifest.json``.
        snapshot:
            Exact immutable source snapshot supplying valid evidence paths.
        expected_manifest:
            Optional in-memory manifest that on-disk bytes must equal.

    Returns:
        Validated on-disk manifest.

    Raises:
        CorpusValidationError:
            If any whole-corpus invariant is violated.
    """

    if not directory.is_dir() or directory.is_symlink():
        raise CorpusValidationError("corpus staging path must be a real directory")
    entries = tuple(directory.iterdir())
    if any(not entry.is_file() or entry.is_symlink() for entry in entries):
        raise CorpusValidationError("corpus must contain only flat regular files")
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise CorpusValidationError("corpus is missing manifest.json")
    manifest = read_manifest(manifest_path)
    if expected_manifest is not None and manifest != expected_manifest:
        raise CorpusValidationError("on-disk manifest differs from expected manifest")

    _validate_source_identity(manifest, snapshot)
    _validate_document_records(manifest, snapshot)
    expected_files = {
        "manifest.json",
        *(record.filename for record in manifest.documents),
    }
    actual_files = {entry.name for entry in entries}
    if actual_files != expected_files:
        raise CorpusValidationError("corpus file set does not match its manifest")

    for record in manifest.documents:
        raw = (directory / record.filename).read_bytes()
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
    return manifest


def _validate_source_identity(
    manifest: CorpusManifest,
    snapshot: RepositorySnapshot,
) -> None:
    """Require the manifest to identify the supplied immutable snapshot.

    Args:
        manifest:
            Manifest being validated.
        snapshot:
            Supplied source snapshot.

    Raises:
        CorpusValidationError:
            If stable source identities disagree.
    """

    if (
        manifest.project_slug != snapshot.project_slug
        or manifest.repository_url != snapshot.repository_url
        or manifest.requested_ref != snapshot.requested_ref
        or manifest.resolved_commit_sha != snapshot.resolved_commit_sha
        or manifest.source_fingerprint != snapshot.source_fingerprint
    ):
        raise CorpusValidationError("manifest does not identify the supplied snapshot")


def _validate_document_records(
    manifest: CorpusManifest,
    snapshot: RepositorySnapshot,
) -> None:
    """Validate filenames, balanced types, and snapshot evidence membership.

    Args:
        manifest:
            Manifest being validated.
        snapshot:
            Snapshot defining valid evidence paths.

    Raises:
        CorpusValidationError:
            If a record violates a document or evidence invariant.
    """

    filenames = tuple(record.filename for record in manifest.documents)
    if filenames != tuple(sorted(filenames)) or len(filenames) != len(set(filenames)):
        raise CorpusValidationError("manifest filenames must be unique and sorted")
    evidence_members = {file.path for file in snapshot.files}
    project_types: set[str] = set()
    component_ids: set[str] = set()
    for record in manifest.documents:
        pure_name = PurePosixPath(record.filename)
        if pure_name.name != record.filename or not _SAFE_FILENAME.fullmatch(
            record.filename
        ):
            raise CorpusValidationError("manifest contains an unsafe filename")
        try:
            expected_name = build_document_filename(
                manifest.project_slug,
                record.document_type,
                component_id=record.component_id,
            )
        except ValueError as exc:
            raise CorpusValidationError("document identity is invalid") from exc
        if record.filename != expected_name:
            raise CorpusValidationError("document filename does not match its type")
        if not record.evidence_paths:
            raise CorpusValidationError("every document requires evidence")
        if record.evidence_paths != tuple(sorted(set(record.evidence_paths))):
            raise CorpusValidationError("document evidence must be unique and sorted")
        if not set(record.evidence_paths).issubset(evidence_members):
            raise CorpusValidationError("document evidence is outside the snapshot")
        if record.document_type == "component":
            assert record.component_id is not None
            if record.component_id in component_ids:
                raise CorpusValidationError("component ids must be unique")
            component_ids.add(record.component_id)
        else:
            if record.document_type in project_types:
                raise CorpusValidationError("project document types must be unique")
            project_types.add(record.document_type)
    if project_types != {"overview", "architecture", "testing_operations"}:
        raise CorpusValidationError("balanced corpus project documents are incomplete")
    if not component_ids:
        raise CorpusValidationError("balanced corpus requires component documents")
