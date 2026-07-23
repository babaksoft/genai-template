from typing import Annotated

from fastapi import APIRouter, Depends

from genai_template.api.dependencies import get_rag_service
from genai_template.schemas import AnswerRequest, AnswerResponse
from genai_template.services import RagService

router = APIRouter()


@router.post(
    "/answer",
    response_model=AnswerResponse,
)
async def answer(
    request: AnswerRequest,
    rag_service: Annotated[RagService, Depends(get_rag_service)],
) -> AnswerResponse:
    """Generate an answer for a user query.

    Args:
        request:
            Answer generation request.

        rag_service:
            Configured RAG service.

    Returns:
        Generated answer and runtime metrics.
    """

    result = rag_service.answer(request.query)

    return AnswerResponse(
        answer=result.answer,
        metrics=result.metrics,
    )
