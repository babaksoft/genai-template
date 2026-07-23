from pydantic import BaseModel, Field

from genai_template.schemas.run_metrics import RunMetrics


class RagResult(BaseModel):
    """Result model for answer generation."""

    answer: str = Field(
        ...,
        description="Generated answer.",
        examples=["Retrieval-Augmented Generation (RAG) combines..."],
    )

    metrics: RunMetrics = Field(
        ...,
        description="Runtime metrics collected during answer generation.",
    )
