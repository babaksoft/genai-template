"""Answer generation API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from genai_template.api.dependencies import get_rag_service
from genai_template.schemas import AnswerRequest, AnswerResponse
from genai_template.services import IndexNotBuiltError, RagService

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

    try:
        result = rag_service.answer(
            request.query,
            request.experiment_id,
            request.rag_config_id,
        )
    except IndexNotBuiltError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return AnswerResponse(
        answer=result.answer,
        metrics=result.metrics,
    )
