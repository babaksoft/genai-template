"""Run metrics schema."""

from pydantic import BaseModel, ConfigDict, Field


class RunMetrics(BaseModel):
    """Represents metrics collected during a single RAG execution."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    query: str = Field(
        min_length=1,
        description="User query.",
    )

    embedding_model: str = Field(
        min_length=1,
        description="Embedding model used during retrieval.",
    )

    vector_store: str = Field(
        min_length=1,
        description="Vector store used during retrieval.",
    )

    llm_model: str = Field(
        min_length=1,
        description="Language model used for response generation.",
    )

    top_k: int = Field(
        ge=1,
        description="Maximum number of retrieved chunks.",
    )

    retrieved_chunks: int = Field(
        ge=0,
        description="Number of retrieved chunks.",
    )

    best_distance: float | None = Field(
        default=None,
        description="Smallest retrieval distance.",
    )

    worst_distance: float | None = Field(
        default=None,
        description="Largest retrieval distance.",
    )

    context_length: int = Field(
        ge=0,
        description="Context length in characters.",
    )

    prompt_length: int = Field(
        gt=0,
        description="Prompt length in characters.",
    )

    response_length: int = Field(
        gt=0,
        description="Response length in characters.",
    )

    retrieval_time: float = Field(
        ge=0,
        description="Retrieval duration in seconds.",
    )

    generation_time: float = Field(
        ge=0,
        description="Response generation duration in seconds.",
    )

    total_time: float = Field(
        ge=0,
        description="Total request duration in seconds.",
    )
