"""Answer generation API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from genai_template.api.dependencies import get_rag_service
from genai_template.schemas import AnswerRequest, AnswerResponse
from genai_template.services import (
    IndexNotBuiltError,
    IndexUnavailableError,
    RagService,
)

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
        Generated answer, runtime metrics, sources, and citation warnings.
    """

    try:
        result = rag_service.answer(
            request.query,
            request.experiment_id,
            request.rag_config_id,
        )
    except IndexUnavailableError as exc:
        rebuild_endpoint = (
            f"/sources/{exc.status.source_id}/indexes/{exc.status.rag_config_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": str(exc),
                "reason": exc.status.reason.value,
                "action": "rebuild_index",
                "rebuild_endpoint": rebuild_endpoint,
            },
        ) from exc
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
        sources=result.sources,
        citation_warnings=result.citation_warnings,
    )
