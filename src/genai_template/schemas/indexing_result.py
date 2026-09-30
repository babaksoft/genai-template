from pydantic import BaseModel, Field


class IndexingResult(BaseModel):
    """Result model for document indexing.

    Attributes:
        documents_indexed:
            Number of indexed documents.
        chunks_indexed:
            Number of indexed chunks.
        indexing_time:
            Indexing duration in seconds.
        corpus_fingerprint:
            Validated corpus identity for manifest-backed sources.
    """

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

    corpus_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description="Validated manifest corpus fingerprint when available.",
    )
