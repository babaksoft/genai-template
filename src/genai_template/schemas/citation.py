"""Schemas for citation-aware RAG responses."""

from pydantic import BaseModel, Field


class CitationSource(BaseModel):
    """A retrieved source made available to the language model."""

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
