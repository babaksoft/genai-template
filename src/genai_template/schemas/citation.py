"""Schemas for citation-aware RAG responses."""

from pydantic import BaseModel, Field


class CitationSource(BaseModel):
    """A retrieved source made available to the language model.

    Attributes:
        label:
            Request-local source label.
        chunk_id:
            Canonical retrieved chunk identifier.
        document_name:
            Safe generated-document basename.
        section:
            Optional Markdown section containing the chunk.
        content:
            Exact retrieved chunk content.
        distance:
            Retrieval distance for the chunk.
        cited:
            Whether the generated answer cites this source.
        project_slug:
            Stable Portfolio project identifier when provenance is available.
        project_display_name:
            Human-readable Portfolio project name when provenance is available.
        document_type:
            Generated Portfolio document type when provenance is available.
        component_id:
            Component identifier for a Portfolio component document.
        repository_url:
            Normalized repository URL when configured for the project.
        resolved_commit_sha:
            Immutable repository commit used to generate the document.
        corpus_fingerprint:
            Fingerprint of the complete published Portfolio corpus.
        generation_fingerprint:
            Fingerprint of the generated document artifact.
    """

    label: str = Field(..., description="Request-local source label.")
    chunk_id: str = Field(..., description="Canonical retrieved chunk identifier.")
    document_name: str = Field(..., description="Safe document basename.")
    section: str | None = Field(
        default=None,
        description="Optional document section containing the chunk.",
    )
    content: str = Field(..., description="Exact retrieved chunk content.")
    distance: float = Field(..., description="Retrieval distance for the chunk.")
    cited: bool = Field(
        default=False,
        description="Whether the generated answer cites this source.",
    )
    project_slug: str | None = Field(
        default=None,
        description="Stable Portfolio project identifier.",
        exclude_if=lambda value: value is None,
    )
    project_display_name: str | None = Field(
        default=None,
        description="Human-readable Portfolio project name.",
        exclude_if=lambda value: value is None,
    )
    document_type: str | None = Field(
        default=None,
        description="Generated Portfolio document type.",
        exclude_if=lambda value: value is None,
    )
    component_id: str | None = Field(
        default=None,
        description="Component identifier for a Portfolio component document.",
        exclude_if=lambda value: value is None,
    )
    repository_url: str | None = Field(
        default=None,
        description="Normalized project repository URL.",
        exclude_if=lambda value: value is None,
    )
    resolved_commit_sha: str | None = Field(
        default=None,
        description="Immutable repository commit used for generation.",
        exclude_if=lambda value: value is None,
    )
    corpus_fingerprint: str | None = Field(
        default=None,
        description="Fingerprint of the complete published Portfolio corpus.",
        exclude_if=lambda value: value is None,
    )
    generation_fingerprint: str | None = Field(
        default=None,
        description="Fingerprint of the generated document artifact.",
        exclude_if=lambda value: value is None,
    )


class CitationWarning(BaseModel):
    """A non-fatal warning produced while resolving answer citations."""

    code: str = Field(..., description="Machine-readable warning code.")
    message: str = Field(..., description="Human-readable warning description.")
    labels: list[str] = Field(..., description="Citation labels causing the warning.")


class CitationContext(BaseModel):
    """Formatted model context and its ordered citation sources."""

    text: str = Field(..., description="Formatted context supplied to the model.")
    sources: list[CitationSource] = Field(
        ...,
        description="Retrieved sources in context order.",
    )
