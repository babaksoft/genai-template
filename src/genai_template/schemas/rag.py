from pydantic import BaseModel, Field

from genai_template.schemas.retrieved_chunk import RetrievedChunk
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

    retrieved_chunks: list[RetrievedChunk] = Field(
        ...,
        description="Retrieved chunks used for building RAG context.",
    )
