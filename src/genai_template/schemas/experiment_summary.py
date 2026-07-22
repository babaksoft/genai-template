from pydantic import BaseModel, ConfigDict, Field


class ExperimentSummary(BaseModel):
    """Summary statistics for an experiment."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    experiment_name: str = Field(
        min_length=1,
        description="Experiment name.",
    )

    run_count: int = Field(
        ge=0,
        description="Number of recorded runs.",
    )

    average_retrieval_time: float = Field(
        ge=0.0,
        description="Average retrieval time in seconds.",
    )

    average_generation_time: float = Field(
        ge=0.0,
        description="Average response generation time in seconds.",
    )

    average_total_time: float = Field(
        ge=0.0,
        description="Average total request time in seconds.",
    )

    average_retrieved_chunks: float = Field(
        ge=0.0,
        description="Average number of retrieved chunks.",
    )

    average_context_length: float = Field(
        ge=0.0,
        description="Average context length in characters.",
    )

    average_prompt_length: float = Field(
        ge=0.0,
        description="Average prompt length in characters.",
    )

    average_response_length: float = Field(
        ge=0.0,
        description="Average response length in characters.",
    )

    best_distance: float | None = Field(
        default=None,
        ge=0.0,
        description="Best (smallest) retrieval distance across all runs.",
    )

    worst_distance: float | None = Field(
        default=None,
        ge=0.0,
        description="Worst (largest) retrieval distance across all runs.",
    )
