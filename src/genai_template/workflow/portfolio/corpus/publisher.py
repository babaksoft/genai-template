"""Atomic immutable-release publisher for validated Portfolio corpora."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

from pydantic import Field

from genai_template.workflow.portfolio.corpus.manifest import manifest_bytes
from genai_template.workflow.portfolio.corpus.validation import (
    CorpusValidationError,
    validate_corpus_directory,
)
from genai_template.workflow.portfolio.domain.generation import RenderedDocument
from genai_template.workflow.portfolio.domain.manifest import CorpusManifest
from genai_template.workflow.portfolio.domain.snapshot import (
    RepositorySnapshot,
    _ImmutableDomainModel,
)

_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")


class PublicationResult(_ImmutableDomainModel):
    """Paths and reuse status resulting from corpus publication.

    Attributes:
        publication_path:
            Atomically replaced public corpus symlink.
        release_path:
            Immutable release directory selected by the symlink.
        corpus_fingerprint:
            Stable identity naming the immutable release.
        release_reused:
            Whether an identical release already existed.
    """

    publication_path: Path = Field(description="Published corpus symlink path.")
    release_path: Path = Field(description="Immutable release directory path.")
    corpus_fingerprint: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Stable published corpus identity.",
    )
    release_reused: bool = Field(
        description="Whether an identical immutable release was reused."
    )


def publish_corpus(
    *,
    publication_path: Path,
    manifest: CorpusManifest,
    documents: Sequence[RenderedDocument],
    snapshot: RepositorySnapshot,
) -> PublicationResult:
    """Stage, validate, and atomically publish one immutable corpus release.

    Args:
        publication_path:
            Public pointer to replace, conventionally ``data/portfolio``.
        manifest:
            Complete stable manifest for the rendered documents.
        documents:
            Rendered Markdown documents to publish.
        snapshot:
            Exact source snapshot used to validate evidence membership.

    Returns:
        Immutable publication paths and release-reuse status.

    Raises:
        CorpusValidationError:
            If staging, an existing release, or the public target is unsafe.
        OSError:
            If a filesystem operation fails.
    """

    publication_path = publication_path.absolute()
    releases_path = publication_path.parent / ".portfolio-releases"
    releases_path.mkdir(parents=True, exist_ok=True)
    _validate_publication_target(publication_path, releases_path)

    stage_path = Path(tempfile.mkdtemp(prefix=".staging-", dir=releases_path))
    release_path = releases_path / manifest.corpus_fingerprint
    reused = False
    try:
        _write_stage(stage_path, manifest, documents)
        validate_corpus_directory(
            stage_path,
            snapshot,
            expected_manifest=manifest,
        )
        if release_path.exists() or release_path.is_symlink():
            _validate_existing_release(release_path, stage_path, snapshot, manifest)
            reused = True
        else:
            try:
                os.rename(stage_path, release_path)
            except FileExistsError:
                _validate_existing_release(release_path, stage_path, snapshot, manifest)
                reused = True
        _switch_pointer(publication_path, release_path)
    finally:
        if stage_path.exists():
            shutil.rmtree(stage_path)

    return PublicationResult(
        publication_path=publication_path,
        release_path=release_path,
        corpus_fingerprint=manifest.corpus_fingerprint,
        release_reused=reused,
    )


def _write_stage(
    stage_path: Path,
    manifest: CorpusManifest,
    documents: Sequence[RenderedDocument],
) -> None:
    """Write all proposed release files into an unpublished directory.

    Args:
        stage_path:
            Fresh temporary release directory.
        manifest:
            Manifest to write canonically.
        documents:
            Rendered Markdown documents to stage.

    Raises:
        CorpusValidationError:
            If provided documents do not correspond one-to-one with the manifest.
    """

    records = {record.filename: record for record in manifest.documents}
    supplied = {document.filename: document for document in documents}
    if len(supplied) != len(documents) or supplied.keys() != records.keys():
        raise CorpusValidationError("rendered documents do not match the manifest")
    for filename, document in supplied.items():
        content = document.content.encode("utf-8")
        record = records[filename]
        if (
            document.document_type != record.document_type
            or document.component_id != record.component_id
            or document.generation_fingerprint != record.generation_fingerprint
            or document.artifact_hash != record.artifact_hash
            or document.content_hash != record.content_hash
            or document.evidence_paths != record.evidence_paths
            or len(content) != record.byte_size
        ):
            raise CorpusValidationError(
                "rendered document metadata differs from manifest"
            )
        (stage_path / filename).write_bytes(content)
    (stage_path / "manifest.json").write_bytes(manifest_bytes(manifest))


def _validate_publication_target(publication_path: Path, releases_path: Path) -> None:
    """Allow only an absent target or a workflow-managed release symlink.

    Args:
        publication_path:
            Prospective public corpus pointer.
        releases_path:
            Workflow-owned immutable release parent.

    Raises:
        CorpusValidationError:
            If the target is a real entry or an unmanaged symlink.
    """

    if not publication_path.is_symlink():
        if publication_path.exists():
            raise CorpusValidationError(
                "publication target is an existing real file or directory"
            )
        return
    target = Path(os.readlink(publication_path))
    resolved_target = (publication_path.parent / target).absolute()
    if (
        target.is_absolute()
        or resolved_target.parent != releases_path
        or not _FINGERPRINT.fullmatch(resolved_target.name)
    ):
        raise CorpusValidationError("publication target is not a managed symlink")


def _validate_existing_release(
    release_path: Path,
    stage_path: Path,
    snapshot: RepositorySnapshot,
    manifest: CorpusManifest,
) -> None:
    """Validate and byte-compare an immutable release selected for reuse.

    Args:
        release_path:
            Existing fingerprint-named release.
        stage_path:
            Newly staged proposed release.
        snapshot:
            Exact source snapshot used for evidence validation.
        manifest:
            Expected stable manifest.

    Raises:
        CorpusValidationError:
            If the existing release conflicts with the proposed bytes.
    """

    validate_corpus_directory(release_path, snapshot, expected_manifest=manifest)
    stage_names = {entry.name for entry in stage_path.iterdir()}
    release_names = {entry.name for entry in release_path.iterdir()}
    if stage_names != release_names or any(
        (stage_path / name).read_bytes() != (release_path / name).read_bytes()
        for name in stage_names
    ):
        raise CorpusValidationError("existing release conflicts with staged corpus")


def _switch_pointer(publication_path: Path, release_path: Path) -> None:
    """Atomically switch the public relative symlink to an immutable release.

    Args:
        publication_path:
            Public corpus pointer.
        release_path:
            Validated immutable release directory.
    """

    relative_target = os.path.relpath(release_path, publication_path.parent)
    temporary_link = publication_path.parent / (
        f".{publication_path.name}.tmp-{os.getpid()}"
    )
    try:
        os.symlink(relative_target, temporary_link)
        os.replace(temporary_link, publication_path)
    finally:
        if temporary_link.is_symlink():
            temporary_link.unlink()
