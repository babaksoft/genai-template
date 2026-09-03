from datetime import datetime

from pydantic import BaseModel, Field

from genai_template.schemas.run_metrics import RunMetrics


class HealthResponse(BaseModel):
    """Response model for API health checks."""

    status: str = Field(
        ...,
        description="Current health status of the API.",
        examples=["healthy"],
    )


class AnswerRequest(BaseModel):
    """Request model for answer generation."""

    query: str = Field(
        ...,
        min_length=1,
        description="User question submitted to the RAG pipeline.",
        examples=["What is Retrieval-Augmented Generation?"],
    )

    source_id: int = Field(
        ...,
        ge=1,
        description="Identifier of the source used for retrieval.",
        examples=[1],
    )


class AnswerResponse(BaseModel):
    """Response model for answer generation."""

    answer: str = Field(
        ...,
        description="Generated answer.",
        examples=["Retrieval-Augmented Generation (RAG) combines..."],
    )

    metrics: RunMetrics = Field(
        ...,
        description="Runtime metrics collected during answer generation.",
    )


class SourceCandidateResponse(BaseModel):
    """A corpus directory available for ingestion."""

    name: str = Field(..., description="Directory basename used as source name.")


class CreateSourceRequest(BaseModel):
    """Request to ingest one configured corpus directory."""

    directory: str = Field(
        ...,
        min_length=1,
        description="Immediate directory name under the configured corpus root.",
        examples=["product-docs"],
    )


class SourceResponse(BaseModel):
    """Metadata describing an ingested corpus source."""

    id: int = Field(
        ...,
        description="Unique source identifier.",
    )

    name: str = Field(
        ...,
        description="Source name derived from the ingested directory basename.",
    )

    directory: str = Field(
        ...,
        description="Full path to the ingested directory.",
    )

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

    indexed_at: datetime = Field(
        ...,
        description="Timestamp when the source was indexed.",
    )

    indexing_time: float = Field(
        ...,
        ge=0,
        description="Indexing duration in seconds.",
    )
