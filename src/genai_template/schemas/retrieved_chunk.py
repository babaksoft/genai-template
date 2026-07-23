"""Schema representing a retrieved document chunk."""

from __future__ import annotations

from pydantic import BaseModel, Field

from genai_template.schemas.chunk import DocumentChunk


class RetrievedChunk(BaseModel):
    """A document chunk returned by a retrieval operation."""

    chunk: DocumentChunk

    distance: float = Field(
        ...,
        description=(
            "Provider-defined vector distance. Smaller values generally "
            "indicate a closer match."
        ),
    )
