from datetime import datetime

from pydantic import BaseModel, Field

from genai_template.config import RagConfig
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
    """A corpus directory available for registration."""

    name: str = Field(..., description="Directory basename used as source name.")


class CreateSourceRequest(BaseModel):
    """Request to register one configured corpus directory."""

    directory: str = Field(
        ...,
        min_length=1,
        description="Immediate directory name under the configured corpus root.",
        examples=["product-docs"],
    )


class SourceResponse(BaseModel):
    """Metadata describing a registered corpus source."""

    id: int = Field(
        ...,
        description="Unique source identifier.",
    )

    name: str = Field(
        ...,
        description="Source name derived from the registered directory basename.",
    )

    directory: str = Field(
        ...,
        description="Full path to the registered directory.",
    )

    created_at: datetime = Field(..., description="Registration timestamp.")


class IndexBuildResponse(BaseModel):
    """Transient result of rebuilding a deterministic source index."""

    documents_indexed: int = Field(..., ge=0, description="Indexed documents.")
    chunks_indexed: int = Field(..., ge=0, description="Indexed chunks.")
    indexing_time: float = Field(..., ge=0, description="Build duration in seconds.")


class CreateExperimentRequest(BaseModel):
    """Request to create a source-bound experiment."""

    source_id: int = Field(..., ge=1, description="Registered source identifier.")
    name: str = Field(..., min_length=1, max_length=128, description="Display name.")
    description: str | None = Field(
        default=None,
        description="Optional human-readable experiment description.",
    )


class ExperimentResponse(BaseModel):
    """Metadata describing a registered experiment."""

    id: int = Field(..., description="Canonical experiment identifier.")
    source_id: int = Field(..., description="Registered source identifier.")
    name: str = Field(..., description="Experiment display name.")
    description: str | None = Field(..., description="Optional description.")
    created_at: datetime = Field(..., description="Registration timestamp.")


class RagConfigResponse(BaseModel):
    """A registered immutable RAG configuration."""

    id: int = Field(..., description="Canonical configuration identifier.")
    config_fingerprint: str = Field(..., description="Full configuration hash.")
    config: RagConfig = Field(..., description="Validated RAG configuration.")
    created_at: datetime = Field(..., description="Registration timestamp.")
