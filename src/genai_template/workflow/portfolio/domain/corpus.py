from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from genai_template.workflow.portfolio.domain.artifacts import ArtifactKind
from genai_template.workflow.portfolio.domain.base import ImmutableDomainModel


class RenderedDocument(ImmutableDomainModel):
    """One deterministic Markdown document derived from an artifact.

    Attributes:
        filename:
            Safe flat corpus filename.
        document_type:
            Logical type of the rendered document.
        component_id:
            Component unit identifier for component documents.
        generation_fingerprint:
            Generation identity of the source artifact.
        artifact_hash:
            SHA-256 identity of the structured source artifact.
        content:
            Deterministically rendered Markdown text.
        content_hash:
            SHA-256 identity of the UTF-8 Markdown bytes.
        evidence_paths:
            Ordered repository evidence paths represented by the document.
    """

    filename: str = Field(min_length=1, description="Safe flat corpus filename.")
    document_type: ArtifactKind = Field(description="Logical document type.")
    component_id: str = Field(
        min_length=1,
        description="Component unit identifier.",
    )
    generation_fingerprint: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Generation identity of the source artifact.",
    )
    artifact_hash: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 identity of the structured source artifact.",
    )
    content: str = Field(description="Deterministically rendered Markdown text.")
    content_hash: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 identity of the UTF-8 Markdown bytes.",
    )
    evidence_paths: tuple[str, ...] = Field(
        description="Ordered repository evidence paths."
    )


class CorpusBuild(ImmutableDomainModel):
    """Complete in-memory document set awaiting corpus publication.

    Attributes:
        project_slug:
            Stable project identifier.
        source_fingerprint:
            SHA-256 identity of the immutable source snapshot.
        resolved_commit_sha:
            Immutable repository commit used for the build.
        documents:
            Ordered rendered Markdown documents in the build.
    """

    project_slug: str = Field(min_length=1, description="Stable project identifier.")
    source_fingerprint: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 identity of the immutable source snapshot.",
    )
    resolved_commit_sha: str = Field(
        pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$",
        description="Immutable repository commit used for the build.",
    )
    documents: tuple[RenderedDocument, ...] = Field(
        min_length=1,
        description="Ordered rendered Markdown documents in the build.",
    )


class PublicationResult(ImmutableDomainModel):
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


class LoadedCorpus(ImmutableDomainModel):
    """A validated corpus pinned to one immutable filesystem location.

    Attributes:
        release_path:
            Resolved directory used for all reads in this load operation.
        manifest:
            Version-neutral stable corpus provenance.
        documents:
            Ordered normalized document records.
    """

    release_path: Path = Field(description="Pinned validated release directory.")
    manifest: CorpusManifest = Field(
        description="Version-neutral normalized corpus manifest."
    )
    documents: tuple[ManifestDocument, ...] = Field(
        min_length=1,
        description="Ordered normalized document records.",
    )


class ManifestPrompt(ImmutableDomainModel):
    """Stable identity of one prompt template used by the corpus.

    Attributes:
        prompt_id:
            Stable versioned prompt identifier.
        prompt_hash:
            SHA-256 hash of the exact prompt template.
    """

    prompt_id: str = Field(min_length=1, description="Stable prompt identifier.")
    prompt_hash: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 hash of the exact prompt template.",
    )


class ManifestDocument(ImmutableDomainModel):
    """Stable manifest record for one rendered Markdown document.

    Attributes:
        project_slug:
            Slug of the project that owns this document.
        file_name:
            Safe flat Markdown file name.
        document_type:
            Logical type of the rendered document.
        component_id:
            Component identifier for a component document.
        evidence_paths:
            Ordered repository-relative evidence paths.
        generation_fingerprint:
            SHA-256 identity of the structured-generation inputs.
        artifact_hash:
            SHA-256 hash of the validated structured output.
        content_hash:
            SHA-256 hash of the published Markdown bytes.
        byte_size:
            Size of the published Markdown in bytes.
    """

    project_slug: str = Field(
        min_length=1,
        description="Slug of the project that owns this document.",
    )
    file_name: str = Field(min_length=1, description="Safe flat Markdown file name.")
    document_type: ArtifactKind = Field(description="Logical document type.")
    component_id: str = Field(
        min_length=1,
        description="Component identifier for this document.",
    )
    evidence_paths: tuple[str, ...] = Field(
        description="Ordered repository-relative evidence paths."
    )
    generation_fingerprint: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 identity of structured-generation inputs.",
    )
    artifact_hash: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 hash of the validated structured output.",
    )
    content_hash: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 hash of the Markdown bytes.",
    )
    byte_size: int = Field(ge=0, description="Published Markdown size in bytes.")


class ManifestProject(ImmutableDomainModel):
    """Stable source and generation provenance for one v2 corpus project.

    Attributes:
        project_slug:
            Stable configured project identifier.
        project_display_name:
            Human-readable configured project name.
        repository_url:
            Normalized repository URL when available.
        requested_ref:
            Repository ref requested before commit resolution.
        resolved_commit_sha:
            Immutable repository commit used for generation.
        source_fingerprint:
            SHA-256 identity of the selected source snapshot.
        generation_profile:
            Corpus generation profile name.
    """

    project_slug: str = Field(min_length=1, description="Stable project identifier.")
    project_display_name: str = Field(
        min_length=1,
        description="Human-readable project name.",
    )
    repository_url: str | None = Field(
        default=None,
        description="Normalized repository URL when available.",
    )
    requested_ref: str = Field(
        min_length=1,
        description="Repository ref requested before commit resolution.",
    )
    resolved_commit_sha: str = Field(
        pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$",
        description="Immutable repository commit used for generation.",
    )
    source_fingerprint: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 identity of the selected source snapshot.",
    )
    generation_profile: Literal["balanced"] = Field(
        description="Corpus generation profile name."
    )


class CorpusManifest(ImmutableDomainModel):
    """Canonical multi-project published corpus manifest.

    Attributes:
        schema_version:
            Version of this manifest contract.
        prompt_version:
            Version of the configured prompt family.
        prompts:
            Ordered prompt template identities used by generation.
        output_schema_version:
            Version of the structured-output schemas.
        provider:
            Structured-generation provider identifier.
        model:
            Provider model identifier.
        configuration_fingerprint:
            Stable content-affecting generation configuration identity.
        projects:
            Ordered non-empty registry of project provenance.
        documents:
            Ordered records corresponding one-to-one with Markdown files.
        corpus_fingerprint:
            SHA-256 identity of the complete stable corpus.
    """

    schema_version: Literal[1] = Field(
        default=1,
        description="Version of the corpus manifest contract.",
    )
    prompt_version: str = Field(
        min_length=1,
        description="Configured prompt-family version.",
    )
    prompts: tuple[ManifestPrompt, ...] = Field(
        min_length=1,
        description="Ordered prompt template identities used by generation.",
    )
    output_schema_version: str = Field(
        min_length=1,
        description="Validated structured-output schema version.",
    )
    provider: Literal["ollama", "openai"] = Field(
        description="Structured-generation provider identifier."
    )
    model: str = Field(min_length=1, description="Provider model identifier.")
    configuration_fingerprint: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Stable content-affecting configuration identity.",
    )
    projects: tuple[ManifestProject, ...] = Field(
        min_length=1,
        description="Ordered project provenance registry.",
    )
    documents: tuple[ManifestDocument, ...] = Field(
        min_length=1,
        description="Ordered project-owned published-document records.",
    )
    corpus_fingerprint: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 identity of the complete stable corpus.",
    )
