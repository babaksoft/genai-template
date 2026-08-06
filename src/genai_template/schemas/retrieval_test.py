from pydantic import BaseModel, Field


class RetrievalTest(BaseModel):
    """Define a single retrieval evaluation case.

    Attributes:
        question:
            User question to evaluate.

        expected_documents:
            Document filenames expected to contain relevant information.
    """

    question: str = Field(
        min_length=1,
        description="User question to evaluate.",
    )

    expected_documents: list[str] = Field(
        min_length=1,
        description="Document filenames expected to contain relevant information.",
    )
