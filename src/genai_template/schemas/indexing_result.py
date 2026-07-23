from pydantic import BaseModel, Field


class IndexingResult(BaseModel):
    """Result model for document indexing."""

    documents_indexed: int = Field(
        ...,
        ge=0,
        description="Number of indexed documents.",
    )

    chunks_indexed: int = Field(
        ...,
        ge=0,
        description="Number of indexed chunks.",
    )

    indexing_time: float = Field(
        ...,
        ge=0,
        description="Indexing duration in seconds.",
    )
