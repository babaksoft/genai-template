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
