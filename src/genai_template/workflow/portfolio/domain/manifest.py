"""Immutable contracts for a published Portfolio corpus manifest."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from genai_template.workflow.portfolio.domain.generation import ArtifactKind
from genai_template.workflow.portfolio.domain.snapshot import _ImmutableDomainModel


class ManifestPrompt(_ImmutableDomainModel):
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


class ManifestDocument(_ImmutableDomainModel):
    """Stable manifest record for one rendered Markdown document.

    Attributes:
        filename:
            Safe flat Markdown filename.
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

    filename: str = Field(min_length=1, description="Safe flat Markdown filename.")
    document_type: ArtifactKind = Field(description="Logical document type.")
    component_id: str | None = Field(
        default=None,
        description="Component identifier when this is a component document.",
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

    @model_validator(mode="after")
    def validate_component_identity(self) -> ManifestDocument:
        """Require component identity only on component documents.

        Returns:
            The validated document record.

        Raises:
            ValueError:
                If document type and component identity disagree.
        """

        if self.document_type == "component" and self.component_id is None:
            raise ValueError("component manifest records require a component id")
        if self.document_type != "component" and self.component_id is not None:
            raise ValueError("project manifest records cannot have a component id")
        return self


class CorpusManifest(_ImmutableDomainModel):
    """Canonical stable provenance for one complete published corpus.

    Attributes:
        manifest_schema_version:
            Version of this manifest contract.
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
        documents:
            Ordered records corresponding one-to-one with Markdown files.
        corpus_fingerprint:
            SHA-256 identity of the manifest projection and Markdown bytes.
    """

    manifest_schema_version: Literal[1] = Field(
        default=1,
        description="Version of the corpus manifest contract.",
    )
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
    documents: tuple[ManifestDocument, ...] = Field(
        min_length=1,
        description="Ordered published-document records.",
    )
    corpus_fingerprint: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 identity of the complete stable corpus.",
    )
